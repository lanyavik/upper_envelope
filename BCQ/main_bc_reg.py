import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, BC_reg


def bc_reg_learn(env_fn, env_name="Hopper-v2", seed=0, buffer_type="Robust", buffer_seed=0, buffer_size='100K',
			  eval_freq=float(1e3), max_timesteps=float(1e6), lr=1e-3, wd=0,
			  logger_kwargs=dict()):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	file_name = "BCreg_%s_%s" % (env_name, seed)
	buffer_name = "%s_%s_%s_%s" % (buffer_type, env_name, buffer_seed, buffer_size)
	print
	("---------------------------------------")
	print
	("Settings: " + file_name)
	print
	("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(env_name)

	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Initialize policy
	policy = BC_reg.BC_reg(state_dim, action_dim, max_action, lr=lr, wd=wd)

	# Load buffer
	replay_buffer = utils.SARSAReplayBuffer()
	replay_buffer.load(buffer_name)

	evaluations = []
	episode_num = 0
	done = True

	training_iters, epoch = 0, 0
	while training_iters < max_timesteps:
		epoch += 1
		pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), logger=logger)

		avgtest_reward = evaluate_policy(policy, env)
		training_iters += eval_freq

		evaluations.append(avgtest_reward)
		np.save("./results/" + file_name, evaluations)

		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', with_min_and_max=True)
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
	parser.add_argument("--env_name", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="FinalSigma0.5")				# Prepends name to filename.
	parser.add_argument("--buffer_size", default="100K")
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment 
	parser.add_argument('--exp_name', type=str, default='bc')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_reg_learn(lambda: gym.make(args.env), env_name=args.env_name, seed=args.seed, buffer_type=args.buffer_type,
			     buffer_size=args.buffer_size, eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
			     logger_kwargs=logger_kwargs)

