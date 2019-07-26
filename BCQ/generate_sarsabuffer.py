import gym
import numpy as np
import torch
import argparse
import os

import utils
import DDPG


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_size", default=1e5, type=float)		# Max time steps to run environment for
	parser.add_argument("--noise1", default=0.1, type=float)			# Probability of selecting random action
	parser.add_argument("--noise2", default=0.1, type=float)			# Std of Gaussian exploration noise
	args = parser.parse_args()

	file_name = "DDPG_%s_%s" % (args.env_name, str(args.seed))
	buffer_name = "ExpertN%sN%s_%s_%s" % (str(args.noise1), str(args.noise2), args.env_name, str(args.seed))
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("---------------------------------------")

	from spinup.utils.logx import EpochLogger
	from spinup.utils.run_utils import setup_logger_kwargs
	logger_kwargs = setup_logger_kwargs('BufferQuality_'+buffer_name, args.seed)
	"""set up logger"""
	logger = EpochLogger(**logger_kwargs)
	logger_save_freq = 10

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	env = gym.make(args.env_name)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = int(env.action_space.high[0])

	# Initialize and load policy
	policy = DDPG.DDPG(state_dim, action_dim, max_action)
	policy.load(file_name, "./pytorch_models")

	# Initialize buffer
	replay_buffer = utils.SARSAReplayBuffer()
	
	total_timesteps = 0
	episode_num = 0
	done = True 

	while total_timesteps < args.buffer_size:
		
		if done: 

			if total_timesteps != 0:
				logger.store(EpLen=episode_timesteps, EpRet=episode_reward)
			if (episode_num+1) % logger_save_freq == 0:
				logger.log_tabular('TotalEnvInteracts', total_timesteps)
				logger.log_tabular('Episode', episode_num)
				logger.log_tabular('EpRet', with_min_and_max=True)
				logger.log_tabular('EpLen', with_min_and_max=True)
				logger.dump_tabular()
			
			# Reset environment
			new_obs = env.reset()
			done = False
			action = 'None'
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Add noise to actions
		if np.random.uniform(0, 1) < args.noise1:
			new_action = env.action_space.sample()
		else:
			new_action = policy.select_action(np.array(new_obs))
			if args.noise2 != 0: 
				new_action = (new_action + np.random.normal(0, args.noise2, size=env.action_space.shape[0]))\
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


	# Save final buffer
	replay_buffer.save(buffer_name)
