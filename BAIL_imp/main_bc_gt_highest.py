import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL_imp import utils, bail_training

# check directory
if os.getcwd().find('lanya') == -1:
	os.chdir("/gpfsnyu/scratch/xc1305")
print('data directory', os.getcwd())
# check pytorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on device:", device)

def bc_gthigh_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5_env_0_1000K", buffer_seed=0,
					gamma=0.99, ue_rollout=1000, augment_mc='gain', eval_freq=500, max_timesteps=int(2e5),
					lr=1e-3, wd=0, P=0.3, batch_size=1000,
					logger_kwargs=dict()):

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	file_name = "BCgthigh_%s_%s_%s" % (buffer_type, env_set, seed)
	setting_name = "%s_r%s_g%s" % (buffer_type.replace('env', env_set), ue_rollout, gamma)
	print("---------------------------------------")
	print
	("Task: " + file_name)
	print("Evaluate Policy every", eval_freq * batch_size / 1e6,
		  'epoches; Total', max_timesteps * batch_size / 1e6, 'epoches')
	print("---------------------------------------")

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
	policy = bail_training.BC(state_dim, action_dim, max_action, lr=lr, wd=wd)

	# Load buffer
	if 'sac' in buffer_type:
		replay_buffer = utils.BEAR_ReplayBuffer()
		desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
		buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
		replay_buffer.load(buffer_name)
		buffer_name += '_1000K'
		setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
	elif 'Final' in buffer_type or 'sigma' in buffer_type:
		replay_buffer = utils.ReplayBuffer()
		buffer_name = buffer_type.replace('env', env_set)
		replay_buffer.load(buffer_name)
	elif 'optimal' in buffer_type:
		buffer_name = buffer_type + "_" + env_set + "_" + str(buffer_seed)
		setting_name = buffer_type + "_" + env_set + "_" + str(buffer_seed)
		replay_buffer = utils.ReplayBuffer()
		replay_buffer.load(buffer_name)
	else:
		raise FileNotFoundError('! Unknown type of dataset %s' % buffer_type)

	setting_name += '_Gain' if augment_mc == 'gain' else '_Gt'
	returns = np.load('./results/ueMC_%s.npy' % setting_name, allow_pickle=True).squeeze()
	print('Load mc returns type', augment_mc, 'with gamma:', gamma, 'rollout length:', ue_rollout)

	selected_buffer, selected_len, border = select_batch_gt(replay_buffer, returns, select_percentage=P)


	print('-- Policy train starts --')
	# Initialize policy
	policy = bail_training.BC(state_dim, action_dim, max_action, lr=lr, wd=wd)

	episode_num = 0
	done = True

	training_iters, epoch = 0, 0
	while training_iters < max_timesteps:
		epoch += eval_freq * batch_size / 1e6
		pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), batch_size=batch_size, logger=logger)

		avgtest_reward = evaluate_policy(policy, test_env)
		training_iters += eval_freq

		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', average_only=True)

		logger.dump_tabular()

def select_batch_gt(replay_buffer, returns, select_percentage):

	returns = torch.from_numpy(returns).to(device)
	increasing_returns, increasing_returns_indices = torch.sort(returns)
	gt_bor_ind = increasing_returns_indices[-int(select_percentage*returns.shape[0])]
	gt_border = returns[gt_bor_ind]

	'''begin selection'''
	selected_buffer = utils.ReplayBuffer()
	for i in range(returns.shape[0]):
		ret = returns[i]
		if ret >= gt_border:
			obs, _, act, _, _ = replay_buffer.index(i)
			selected_buffer.add((obs, None, act, None, None))

	initial_len, selected_len = replay_buffer.get_length(), selected_buffer.get_length()
	print(selected_len, '/', initial_len, 'selecting ratio:', selected_len/initial_len)

	#selection_info = 'highgt'
	#selection_info += setting_name
	#selection_info += '_gtbor%.2f_len%s' % (gt_border, selected_len)
	#selected_buffer.save(selection_info)

	return (selected_buffer, selected_len, gt_border)


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
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bc_gt_high')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_gthigh_learn(seed=args.seed, eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
					logger_kwargs=logger_kwargs)
