import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL_imp import utils, BCQ_bl

if os.getcwd().find('lanya') == -1:
	os.chdir("/gpfsnyu/scratch/xc1305")
print('data directory', os.getcwd())

def bcq_learn(env_set="Hopper-v2", seed=0, buffer_type="sacpolicy_env_stopcrt_2_det_bear",
			  batch_size=100, eval_freq=float(5e3), max_timesteps=float(2e6), lr=1e-3,
			  logger_kwargs=dict()):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    """set up logger"""
    global logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    file_name = "BCQ_%s_%s" % (env_set, seed)

    print("---------------------------------------")
    print
    ("Task: " + file_name)
    print("Evaluate Policy every", eval_freq * batch_size / 1e6,
          'epoches; Total', max_timesteps * batch_size / 1e6, 'epoches')
    print("---------------------------------------")

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

    # Initialize policy
    policy = BCQ_bl.BCQ(state_dim, action_dim, max_action, lr=lr)

    # Load buffer
    if 'sac' in buffer_type:
        replay_buffer = utils.BEAR_ReplayBuffer()
        desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
        buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
        replay_buffer.load(buffer_name)
        buffer_name += '_1000K'
        #setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
    elif 'FinalSigma' in buffer_type:
        replay_buffer = utils.ReplayBuffer()
        buffer_name = buffer_type.replace('env', env_set)
        replay_buffer.load(buffer_name)
    elif 'optimal' in buffer_type:
        buffer_name = buffer_type.replace('env', env_set)
        setting_name = buffer_name
        setting_name += 'noaug' if not (augment_mc) else ''
        replay_buffer = utils.ReplayBuffer()
        replay_buffer.load(buffer_name)
    else:
        raise FileNotFoundError('! Unknown type of dataset %s' % buffer_type)


    training_iters, epoch = 0, 0
    while training_iters < max_timesteps:
        epoch += eval_freq * batch_size / 1e6
        pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), batch_size=batch_size, logger=logger)

        avgtest_reward = evaluate_policy(policy, test_env)
        training_iters += eval_freq


        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('AverageTestEpRet', avgtest_reward)
        logger.log_tabular('TotalSteps', training_iters)
        logger.log_tabular('QLoss', average_only=True)
        logger.log_tabular('Q1Vals', with_min_and_max=True)
        logger.log_tabular('Q2Vals', with_min_and_max=True)
        logger.log_tabular('ActLoss', with_min_and_max=True)
        logger.dump_tabular()



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
    parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
    args = parser.parse_args()

    EXP_NAME = 'bcq_bl_local'
    logger_kwargs = setup_logger_kwargs(EXP_NAME, 0)

    bcq_learn(env_set=args.env_set, seed=args.seed,
              #eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
              logger_kwargs=logger_kwargs)

