import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, BC_ue_border_clip
from spinup.algos.ue.MC_UE import plot_envelope_with_clipping

from spinup.algos.ue.models.mlp_critic import Value


def bc_ue_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
                cut_buffer_size='1000K', ue_seed_list=[1], gamma=0.99, ue_rollout=1000, ue_loss_k=10000,
				clip_ue=None, detect_interval=10000,
			    eval_freq=float(500), max_timesteps=float(1e5), lr=1e-3, wd=0, border=0.9,
			    logger_kwargs=dict()):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	rollout_list = [None, 1000, 200, 100, 10]
	k_list = [10000, 1000, 100, 100000, 50000, 5000]

	file_name = "BCueclip_%s_%s" % (env_set, seed)
	buffer_name = "%s_%s_%s" % (buffer_type, env_set, buffer_seed)
	setting_name = "%s_%s_r%s_g%s" % (buffer_name, cut_buffer_size, ue_rollout, gamma)


	print
	("---------------------------------------")
	print
	("Settings: " + file_name)
	print
	("---------------------------------------")


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

	# Load buffer
	replay_buffer = utils.ReplayBuffer()
	replay_buffer.load(buffer_name + '_' + buffer_size)
	if buffer_size != cut_buffer_size:
		replay_buffer.cut_final(int(cut_buffer_size[:-1]) * 1e3)
	print(replay_buffer.get_length())

	print('buffer setting:', buffer_name + '_' + cut_buffer_size)

	print('clip and selection type:', clip_ue)
	env_bs_dic = {'Hopper-v2': [4, 4], 'Walker2d-v2': [3, 5], 'HalfCheetah-v2': [1, 1]}
	if clip_ue is None:
		best_ue_seed = env_bs_dic[env_set][buffer_seed]
		C = None
	elif clip_ue == "s-auto":
		best_ue_seed = env_bs_dic[env_set][buffer_seed]
		print('-- Do clipping on the selected envelope --')
		C, _ = get_ue_clipping_info(best_ue_seed, ue_loss_k, detect_interval, setting_name, state_dim,\
			buffer_info=buffer_name + '_' + cut_buffer_size, ue_setting='[k=%s_MClen=%s_gamma=%s'%(ue_loss_k, ue_rollout, gamma))
	elif clip_ue == "f-auto":
		print('-- Do clipping on each envelope --')
		ues_info = dict()
		for ue_seed in ue_seed_list:
			ues_info[ue_seed] = get_ue_clipping_info(ue_seed, ue_loss_k, detect_interval, setting_name, state_dim,\
			buffer_info=buffer_name + '_' + cut_buffer_size, ue_setting='[k=%s_MClen=%s_gamma=%s'%(ue_loss_k, ue_rollout, gamma))
		print('Auto clipping info:', ues_info)
		clipping_val_list, clipping_loss_list = tuple(map(list, zip(*ues_info.values())))
		sele_idx = int(np.argmin(np.array(clipping_loss_list)))
		best_ue_seed = ue_seed_list[sele_idx]
		C = clipping_val_list[sele_idx]


	print("Best UE", best_ue_seed, "Clipping value: ", C)

	print('-- Policy train starts --')
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	print('Load best envelope from', './results/ueMC_%s_Gt.npy' % setting_name)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(best_ue_seed, ue_loss_k))))
	print('Load best envelope from', '%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(best_ue_seed, ue_loss_k)))
	print('with testing MClength:', ue_rollout, 'training loss ratio k:', ue_loss_k)

	#plot_envelope(upper_envelope, states, actions, gts, buffer_name, seed)





	# Initialize policy
	policy = BC_ue_border_clip.BC_ue(state_dim, action_dim, max_action, lr=lr, wd=wd, ue_valfunc=upper_envelope, mc_rets=gts)

	episode_num = 0
	done = True

	training_iters, epoch = 0, 0
	while training_iters < max_timesteps:
		epoch += 1
		pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), border=border, logger=logger, C=C)

		avgtest_reward = evaluate_policy(policy, test_env)
		training_iters += eval_freq


		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('Loss', average_only=True)
		logger.log_tabular('SVal', with_min_and_max=True)
		logger.log_tabular('UpSize', with_min_and_max=True)
		logger.dump_tabular()


def get_ue_clipping_info(ue_seed, ue_loss_k, detect_interval, setting_name, state_dim, buffer_info, ue_setting):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_info, allow_pickle=True)
	returns = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(
		torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (ue_seed, ue_loss_k))))

	clipping_val, clipping_loss = plot_envelope_with_clipping(upper_envelope, states, returns, buffer_info+ue_setting, ue_seed,
								  hyper_default=True, k_val=ue_loss_k, S=detect_interval)

	return clipping_val, clipping_loss

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
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="FinalSigma0.5")				# Prepends name to filename.
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bc_ue_b')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_ue_learn(env_set=args.env_name, seed=args.seed, buffer_type=args.buffer_type,
                eval_freq=args.eval_freq,
                max_timesteps=args.max_timesteps,
                logger_kwargs=logger_kwargs)
