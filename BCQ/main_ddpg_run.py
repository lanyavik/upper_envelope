import numpy as np
import torch
import gym
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, DDPG_col


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

	return avg_reward


# Shortened version of code originally found at https://github.com/sfujim/TD3
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
	parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
	parser.add_argument("--expl_noise", default=0.5, type=float)  # Std of Gaussian exploration noise
	parser.add_argument("--cut_buffer_size", default=1e5, type=float)
	args = parser.parse_args()


	file_name = "DDPG_%s_%s" % (args.env_name, str(args.seed))
	buffer_name = "FinalSigma%s_%s_%s_%sK" % (str(args.expl_noise), args.env_name, str(args.seed),
										   str(int(args.cut_buffer_size/1e3)))
	exp_name = "ddpg_collection_%s_steps%s_sigma%s_%s" \
			   % (args.env_name, str(args.max_timesteps), str(args.expl_noise), str(args.seed))
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("Save Buffer as: " + buffer_name)
	print ("---------------------------------------")

	if not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")

	logger_kwargs = setup_logger_kwargs(exp_name, args.seed)
	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	env = gym.make(args.env_name)
	test_env = gym.make(args.env_name)

	# Set seeds
	'''for algos with environment interacts we also have to seed env.action_space'''
	env.seed(args.seed)
	test_env.seed(args.seed)
	env.action_space.np_random.seed(args.seed)
	test_env.action_space.np_random.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy and buffer
	policy = DDPG_col.DDPG(state_dim, action_dim, max_action)
	replay_buffer = utils.SARSAReplayBuffer()
	
	total_timesteps = 0
	episode_num = 0
	done = True 

	while total_timesteps < args.max_timesteps:
		
		if done: 

			if total_timesteps != 0:
				policy.train(replay_buffer, episode_timesteps)

				avgtest_reward = evaluate_policy(policy, test_env, eval_episodes=10)


				logger.log_tabular('Episode', episode_num)
				logger.log_tabular('AverageTestEpRet', avgtest_reward)
				logger.log_tabular('TotalSteps', total_timesteps)
				logger.log_tabular('EpRet', episode_reward)
				logger.log_tabular('EpLen', episode_timesteps)
				logger.dump_tabular()


			# Reset environment
			new_obs = env.reset()
			done = False
			action = 'None'
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			new_action = env.action_space.sample()
		else:
			new_action = policy.select_action(np.array(new_obs))
			if args.expl_noise != 0:
				new_action = (new_action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0]))\
							  .clip(env.action_space.low, env.action_space.high)

		# Perform new action!!!
		new_obs_NEXT, reward_NEXT, done, _ = env.step(new_action)
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

		# Store data in replay buffer
		if episode_timesteps != 0:
			replay_buffer.add((obs, new_obs, action, new_action, reward, done_bool))
		obs = new_obs
		new_obs, reward = new_obs_NEXT, reward_NEXT
		action = new_action
		episode_reward += reward_NEXT


		episode_timesteps += 1
		total_timesteps += 1
		
	# Save final policy
	policy.save("%s" % (file_name), directory="./pytorch_models")
	# Save final buffer
	replay_buffer.cut_final(args.cut_buffer_size)
	replay_buffer.save(buffer_name)

