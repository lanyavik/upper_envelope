import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, BC_ue_border_perturb_c, BC_ue_border_perturb_5, BC_ue_border_perturb_e

from spinup.algos.ue.models.mlp_critic import Value, QNet


def bc_ue_ptb_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
                cut_buffer_size='1000K',
                mcue_seed=1, qloss_k=10000, qgt_seed=0, qlearn_type='learn_all_data', border=0.75, clip=0.85,
                update_type='e',
			    eval_freq=float(1e3), max_timesteps=float(1e6), lr=1e-3, lag_lr=1e-3, search_lr=3e-2, wd=0,
				epsilon_base=1,
			    logger_kwargs=dict()):

    """parameters |max_timesteps|, |eval_freq|:
       for BC_ue_border_perturb_c, Totalsteps means the number of minibatch updates (default batch size=100)
       for BC_ue_border_perturb_5,
       for BC_ue_border_perturb_e, Totalsteps means the number of updates on each datapoint, i.e., a step is
                                   an iteration of one optimization step on each data in the buffer"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    """set up logger"""
    global logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    file_name = "BCue_per_e_%s_%s" % (env_set, seed)
    buffer_name = "%s_%s_%s_%s" % (buffer_type, env_set, buffer_seed, buffer_size)
    setting_name = "%s_r%s_g%s" % (buffer_name, 1000, 0.99)
    print
    ("---------------------------------------")
    print
    ("Settings: " + setting_name)
    print
    ("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(env_set)
    test_env = gym.make(env_set)

    # Set seeds
    env.seed(seed)
    test_env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.action_space.np_random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    action_range = float(env.action_space.high[0]) - float(env.action_space.low[0])
    print('env', env_set, 'action range:', action_range)

    # Print out config used in MC Upper Envelope training
    rollout_list = [None, 1000, 200, 100, 10]
    k_list = [10000, 1000, 100]
    print('testing MClength:', rollout_list[mcue_seed % 10])
    print('Training loss ratio k:', k_list[mcue_seed // 10])

    selection_info = 'ue_border%s' % border
    selection_info += '_clip%s' % clip if clip is not None else ''
    print('selection_info:', selection_info)
    # Load the ue border selected buffer
    selected_buffer = utils.SARSAReplayBuffer()
    if buffer_size != cut_buffer_size:
        buffer_name = buffer_name + '_cutfinal' + cut_buffer_size

    selected_buffer.load(selection_info +'_'+ buffer_name)
    buffer_length = selected_buffer.get_length()
    print(buffer_length)
    print('buffer setting:', selection_info +'_'+ buffer_name)



    # Load the Q net trained with regression on Gts
    # And Load the corresponding Gts to the selected buffer
    selected_gts = np.load('./results/sele%s_ueMC_%s_Gt.npy' % (selection_info, setting_name), allow_pickle=True)

    if qlearn_type == 'learn_all_data':
        verbose_qnet = 'alldata_qgts%s' % qgt_seed +'lok=%s' % qloss_k

    elif qlearn_type == 'learn_border_data':
        verbose_qnet = 'uebor%s_qgts%s' % (border, qgt_seed) if clip is None \
                       else 'uebor%s_clip%s_qgts%s' % (border, clip, qgt_seed)
        verbose_qnet += 'lok=%s' % qloss_k
    else: raise ValueError

    print('verbose_qnet:', verbose_qnet)

    Q_from_gt = QNet(state_dim, action_dim, activation='relu')
    Q_from_gt.load_state_dict(torch.load('%s/%s_Qgt.pth' % ("./pytorch_models", setting_name+'_'+verbose_qnet)))
    print('load Qnet from', '%s/%s_UE.pth' % ("./pytorch_models", setting_name))

    # choose the epsilon plan for the constraints
    if update_type == 'c':
        epsilon = epsilon_plan(epsilon_base, action_range, selected_buffer, selected_gts, Q_from_gt, device,\
                               plan='common')
    else:
        epsilon = torch.FloatTensor([epsilon_base])
        print('one epsilon:', epsilon)

    print('policy train starts --')

    '''Initialize policy of the update type'''
    print("Updating approach: BC_ue_border_perturb_%s" % update_type)
    if update_type == "c":
        policy = BC_ue_border_perturb_c.BC_ue_perturb(state_dim, action_dim, max_action,\
                     lr=lr, lag_lr=lag_lr, wd=wd, num_lambda=buffer_length, Q_from_gt=Q_from_gt )
    elif update_type == "5":
        policy = BC_ue_border_perturb_5.BC_ue_perturb(state_dim, action_dim, max_action, \
                                                      lr=lr, lag_lr=lag_lr, wd=wd, Q_from_gt=Q_from_gt)
    elif update_type == "e":
        policy = BC_ue_border_perturb_e.BC_ue_perturb(state_dim, action_dim, max_action, \
                                                      lr=lr, wd=wd, Q_from_gt=Q_from_gt)
        policy.train_a_tilda(selected_buffer, max_updates=50, search_lr=search_lr, epsilon=epsilon)

    episode_num = 0
    done = True

    training_iters, epoch = 0, 0
    while training_iters < max_timesteps:
        epoch += 1
        if update_type == 'e':
            pol_vals = policy.behavioral_cloning(iterations=int(eval_freq), logger=logger)
        else: # "5" and "c"
            pol_vals = policy.train(selected_buffer, iterations=int(eval_freq), epsilon=epsilon, logger=logger)

        avgtest_reward = evaluate_policy(policy, test_env)
        training_iters += eval_freq


        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('AverageTestEpRet', avgtest_reward)
        logger.log_tabular('TotalSteps', training_iters)

        if update_type == 'c':
            logger.log_tabular('BCLoss', average_only=True)
            logger.log_tabular('ActorLoss', average_only=True)
            logger.log_tabular('LambdaMax', average_only=True)
            logger.log_tabular('LambdaMin', average_only=True)
            logger.log_tabular('ConstraintViolated', with_min_and_max=True)
        elif update_type == '5':
            logger.log_tabular('BCLoss', average_only=True)
            logger.log_tabular('ActorLoss', average_only=True)
            logger.log_tabular('Lambda', average_only=True)
            logger.log_tabular('ConstraintViolatedValue', average_only=True)

        elif update_type == 'e':
            logger.log_tabular('BCLoss', average_only=True)

        logger.dump_tabular()


def epsilon_plan(epsilon_base, action_range, selected_buffer, selected_gts, Q_from_gt, device,
                 discount=0.99, plan=None):
    epsilon_base = torch.FloatTensor([epsilon_base]).unsqueeze(dim=0)
    print(epsilon_base.size())

    if plan == 'common':
        epsilon = epsilon_base * torch.ones(selected_gts.shape)
    elif plan == 'inv_proportional_to_normalized_mcrets':
        selected_gts = torch.FloatTensor(selected_gts).unsqueeze(dim=0)
        print(selected_gts.size())
        selected_gts -= selected_gts.min()
        selected_gts = selected_gts / selected_gts.std()
        epsilon = torch.clamp( epsilon_base * selected_gts.pow(-1),
                               epsilon_base.item()*1e-1, epsilon_base.item()*1e1)
    elif plan == 'inv_proportional_to_bellman_residuals':
        Q_from_gt = Q_from_gt.to(device)
        bellman_res = []
        for ind in range(selected_buffer.get_length()):
            state, next_state, action, _, reward, done = selected_buffer.index(ind)

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.FloatTensor(action).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            next_action = torch.FloatTensor(next_action).unsqueeze(0).to(device)
            reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            done = torch.FloatTensor([1 - done]).unsqueeze(0).to(device)

            err = reward + discount * done * Q_from_gt(torch.cat([next_state, next_action], 1)) \
                  - Q_from_gt(torch.cat([state, action], 1))
            bellman_res.append(np.absolute(err.detach().cpu().numpy()))

        bellman_res=np.asarray(bellman_res)
        epsilon = epsilon_base * bellman_res.pow(-1)

    else:
        print('Invalid Plan!')
        raise ValueError

    print('eps mean', epsilon.mean(), 'eps std', epsilon.std(), 'eps max', epsilon.max(), 'eps min', epsilon.min())
    print('eps size', epsilon.size())
    return epsilon

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=10):
    tol_reward = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            tol_reward += reward

    avg_reward = tol_reward / eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward



if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_set", default="Hopper-v2")				# OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="MedSACtest1000")				# Prepends name to filename.
    parser.add_argument("--buffer_size", default="1000K")
    parser.add_argument("--cut_buffer_size", default="1000K")
    parser.add_argument("--buffer_seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
    parser.add_argument('--exp_name', type=str, default='bc_ue_ptb')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    bc_ue_ptb_learn(env_set=args.env_set, seed=args.seed, buffer_type=args.buffer_type,
                      buffer_size=args.buffer_size, cut_buffer_size=args.cut_buffer_size, buffer_seed=args.buffer_seed,
                      eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
                      logger_kwargs=logger_kwargs)
