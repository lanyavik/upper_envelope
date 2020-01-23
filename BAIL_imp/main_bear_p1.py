import gym
import numpy as np
import torch
import argparse
import os

from spinup.algos.BAIL_imp import utils, BEAR_bl
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
#import point_mass

if os.getcwd().find('lanya') == -1:
	os.chdir("/gpfsnyu/scratch/xc1305")
print('data directory', os.getcwd())

def bear_learn(algo_name='BEAR', version='0', env_set="Hopper-v2", seed=0, buffer_type='FinalSigma0.5_env_0_1000K',
		lamda=0.0, threshold=0.05, use_bootstrap=False, bootstrap_dim=4, delta_conf=0.1,
		mode='auto', kernel_type='laplacian', num_samples_match=5, mmd_sigma=10.0,
		lagrange_thresh=10.0, distance_type='MMD',
		use_ensemble_variance=False, use_behaviour_policy=False,
		num_random=10, margin_threshold=10,
		batch_size=100, max_timesteps=2e6, eval_freq=5e3, logger_kwargs=dict()):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    """set up logger"""
    global logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    exp_name = algo_name + "_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_0.1" % (env_set, str(seed), str(version), str(lamda), str(threshold), str(use_bootstrap), str(mode),\
         str(kernel_type), str(num_samples_match), str(mmd_sigma), str(lagrange_thresh), str(distance_type), str(use_behaviour_policy), str(num_random))
    #buffer_name = buffer_name
    print ("---------------------------------------")
    print ("Settings: " + exp_name)
    print("Evaluate Policy every", eval_freq * batch_size / 1e6,
          'epoches; Total', max_timesteps * batch_size / 1e6, 'epoches')
    print("---------------------------------------")

    # if not os.path.exists("./results"):
    # 	os.makedirs("./results")

    #if env_set == 'Multigoal-v0':
    #	env = point_mass.MultiGoalEnv(distance_cost_coeff=10.0)
    #else:
        #env = gym.make(env_set)
    env = gym.make(env_set)
    test_env = gym.make(env_set)
    env.seed(seed)
    test_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print (state_dim, action_dim)
    print ('Max action: ', max_action)


    if algo_name == 'BEAR':
        policy = BEAR_bl.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
            version=version,
            lambda_=float(lamda),
            threshold=float(threshold),
            mode=mode,
            num_samples_match=num_samples_match,
            mmd_sigma=mmd_sigma,
            lagrange_thresh=lagrange_thresh,
            use_kl=(True if distance_type == "KL" else False),
            use_ensemble=use_ensemble_variance,
            kernel_type=kernel_type)
    elif algo_name == 'BEAR_IS':
        policy = BEAR_bl.BEAR_IS(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
            version=version,
            lambda_=float(lamda),
            threshold=float(threshold),
            mode=mode,
            num_samples_match=num_samples_match,
            mmd_sigma=mmd_sigma,
            lagrange_thresh=lagrange_thresh,
            use_kl=(True if distance_type == "KL" else False),
            use_ensemble=use_ensemble_variance,
            kernel_type=kernel_type)

    # Load buffer
    if 'sac' in buffer_type:
        replay_buffer = utils.BEAR_ReplayBuffer()
        desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
        buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
        replay_buffer.load(buffer_name, bootstrap_dim=4)
        buffer_name += '_1000K'
        #setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
    elif 'FinalSigma' in buffer_type:
        replay_buffer = utils.ReplayBuffer()
        buffer_name = buffer_type.replace('env', env_set)
        replay_buffer.load(buffer_name, bootstrap_dim=4)
    else:
        raise FileNotFoundError('! Unknown type of dataset %s' % buffer_type)


    training_iters, epoch = 0, 0
    while training_iters < max_timesteps:
        epoch += eval_freq * batch_size / 1e6
        pol_vals = policy.train(replay_buffer, iterations=int(eval_freq))

        try:
            ret_eval, var_ret, median_ret = evaluate_policy(policy, test_env)
        except:
            print('invalid evaluation!')
            ret_eval, var_ret, median_ret = 0, 0, 0

        training_iters += eval_freq
        logger.log_tabular("TotalSteps", training_iters)
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('AverageTestEpRet', ret_eval)
        logger.log_tabular('VarianceReturn', var_ret)
        logger.log_tabular('MedianReturn', median_ret)
        logger.dump_tabular()


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=10):
	avg_reward = 0.
	all_rewards = []
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		cntr = 0
		while ((not done)):
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward
			cntr += 1
		all_rewards.append(avg_reward)
	avg_reward /= eval_episodes
	for j in range(eval_episodes - 1, 1, -1):
		all_rewards[j] = all_rewards[j] - all_rewards[j-1]

	all_rewards = np.array(all_rewards)
	std_rewards = np.std(all_rewards)
	median_reward = np.median(all_rewards)
	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward, std_rewards, median_reward

def evaluate_policy_discounted(policy, env, eval_episodes=10):
	avg_reward = 0.
	all_rewards = []
	gamma = 0.99
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		cntr = 0
		gamma_t = 1
		while ((not done)):
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += (gamma_t * reward)
			gamma_t = gamma * gamma_t
			cntr += 1
		all_rewards.append(avg_reward)
	avg_reward /= eval_episodes
	for j in range(eval_episodes-1, 1, -1):
		all_rewards[j] = all_rewards[j] - all_rewards[j-1]

	all_rewards = np.array(all_rewards)
	std_rewards = np.std(all_rewards)
	median_reward = np.median(all_rewards)
	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward, std_rewards, median_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")                          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                                      # Sets Gym, PyTorch and Numpy seeds
	#parser.add_argument("--buffer_type", default="Robust")                          # Prepends name to filename.
	parser.add_argument("--eval_freq", default=1e2, type=float)                     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e5, type=float)         # Max time steps to run environment for

	args = parser.parse_args()

	EXP_NAME = 'bear_bl_local'
	logger_kwargs = setup_logger_kwargs(EXP_NAME, 0)

	bear_learn(
		seed=0,
        # eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
	)
