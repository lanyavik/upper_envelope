import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, BC_ue_border
from spinup.algos.ue.PPO_UE import train_upper_envelope, plot_envelope

from spinup.algos.ue.models.mlp_critic import Value


def bc_ue_learn(env_set="Hopper-v2", seed=0, buffer_type="Robust", buffer_seed=1, buffer_size='1000K',
                cut_buffer_size='500K', ue_seed=1, max_ue_trainsteps=1e6,
			    eval_freq=float(1e3), max_timesteps=float(1e6), lr=1e-3, wd=0, border=0.9,
			    logger_kwargs=dict()):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	file_name = "BCue_%s_%s" % (env_set, seed)
	buffer_name = "%s_%s_%s" % (buffer_type, env_set, buffer_seed)
	setting_name = "%s_%s_%s" % (buffer_name, cut_buffer_size, ue_seed)
	print
	("---------------------------------------")
	print
	("Settings: " + file_name)
	print
	("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(env_set)

	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Load buffer
	replay_buffer = utils.SARSAReplayBuffer()
	replay_buffer.load(buffer_name + '_' + buffer_size)
	if buffer_size != cut_buffer_size:
		replay_buffer.cut_final(int(cut_buffer_size[:-1]) * 1e3)
	print(replay_buffer.get_length())

	print('buffer setting:', buffer_name + '_' + cut_buffer_size)
	'''
	# extract (s,a,r) pairs from replay buffer
	length = replay_buffer.get_length()
	print(length)
	states, actions, gts = [], [], []
	for ind in range(length):
		state, _, action, _, _, _ = replay_buffer.index(ind)
		gt =  calculate_mc_ret(replay_buffer, ind)
		states.append(state)
		actions.append(action)
		gts.append(gt)

	print('ue train starts ==')

	states = np.load('./results/ue_%s_S.txt.npy' % file_name, allow_pickle=True)
	actions = np.load('./results/ue_%s_A.txt.npy' % file_name, allow_pickle=True)
	gts = np.load('./results/ue_%s_Gt.txt.npy' % file_name, allow_pickle=True)

	upper_envelope = train_upper_envelope(states, actions, gts, state_dim, device, seed)
	torch.save(upper_envelope.state_dict(), '%s/%s_UE.pth' % ("./pytorch_models", file_name))
	
	print('ue train finished --')
	input('Proceed>')
	'''

	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name)))
	print('load envelope from', '%s/%s_UE.pth' % ("./pytorch_models", setting_name))

	#plot_envelope(upper_envelope, states, actions, gts, buffer_name, seed)

	print('policy train starts --')

	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	rollout_list = [None, 1000, 200, 100, 10]
	k_list = [10000, 1000, 100]
	print('testing MClength:', rollout_list[ue_seed % 10])
	print('Training loss ratio k:', k_list[ue_seed // 10])


	# Initialize policy
	policy = BC_ue_border.BC_ue(state_dim, action_dim, max_action, lr=lr, wd=wd, ue_valfunc=upper_envelope, mc_rets=gts)

	episode_num = 0
	done = True

	training_iters, epoch = 0, 0
	while training_iters < max_timesteps:
		epoch += 1
		pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), border=border, logger=logger)

		avgtest_reward = evaluate_policy(policy, env)
		training_iters += eval_freq


		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', average_only=True)
		logger.log_tabular('SVal', with_min_and_max=True)
		logger.log_tabular('UpSize', with_min_and_max=True)
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

def calculate_mc_ret(replay_buffer, idx, rollout=10, discount=0.99):
	r_length = replay_buffer.get_length()
	o, _, a, _, r, d = replay_buffer.index(idx)
	sampled_policy_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			pass
		else:
			_, _, _, _, r, d = replay_buffer.index(idx + h)
			sampled_policy_est += discount ** h * r

	return np.asarray(sampled_policy_est)



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="FinalSigma0.5")				# Prepends name to filename.
	parser.add_argument("--buffer_size", default="1000K")
	parser.add_argument("--cut_buffer_size", default="1000K")
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bc_ue_b')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_ue_learn(env_set=args.env_set, seed=args.seed, buffer_type=args.buffer_type,
                buffer_size=args.buffer_size, cut_buffer_size=args.cut_buffer_size,
				eval_freq=args.eval_freq, max_timesteps=args.max_timesteps,
                logger_kwargs=logger_kwargs)
