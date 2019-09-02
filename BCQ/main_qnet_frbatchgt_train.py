import gym
import numpy as np
import torch
import argparse
import os
from spinup.algos.BCQ import utils
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.ue.MC_UE_Qnet import reg_qnet_to_batch, plot_envelope

from spinup.algos.ue.models.mlp_critic import Value, QNet


def batch_qnet_train(env_set="Hopper-v2", seed=0,  buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
			 		cut_buffer_size='1000K',
			 		qlearn_type='learn_all_data', vue_seed=1, vue_loss_k=10000, border=0.75, clip=None, qloss_k=10000,
			 		max_q_trainsteps=int(1e6), lr=1e-3, wd=0, discount=0.99, logger_kwargs=dict()):


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())


	buffer_name = "%s_%s_%s_%s" % (buffer_type, env_set, buffer_seed, buffer_size)
	setting_name = "%s_r%s_g%s" % (buffer_name, 1000, 0.99)
	print("---------------------------------------")
	print("Settings: " + setting_name)
	print("---------------------------------------")

	if not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")

	env = gym.make(env_set)

	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])


	# Load buffer
	replay_buffer = utils.SARSAReplayBuffer()
	replay_buffer.load(buffer_name)
	if buffer_size != cut_buffer_size:
		replay_buffer.cut_final(int(cut_buffer_size[:-1]) * 1e3)
		buffer_name = buffer_name + '_cutfinal' + cut_buffer_size
	print(replay_buffer.get_length())

	print('buffer setting:', buffer_name)

	# extract (s,a,r) pairs from replay buffer
	#selected_states, selected_actions, selected_gts \
	#	= select_batch_ue_border(replay_buffer, setting_name, '_s%s_lok%s'%(vue_seed, vue_loss_k),
	#							 buffer_name, state_dim, border=border, clip=clip)

	print('qlearn_type:', qlearn_type)
	if qlearn_type == 'learn_all_data':

		states = np.load('./results/ueMC_%s_S.npy' % buffer_name, allow_pickle=True)
		actions = np.load('./results/ueMC_%s_A.npy' % buffer_name, allow_pickle=True)
		gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)

		verbose_qnet = 'alldata_qgts%s' % seed +'lok=%s' % qloss_k
		Q_from_gt = reg_qnet_to_batch(states, actions, gts, state_dim, action_dim, device, seed,
									  q_learning_rate=lr, q_weight_decay=wd, max_step_num=max_q_trainsteps, k=qloss_k)

		torch.save(Q_from_gt.state_dict(), '%s/%s_Qgt.pth' % ("./pytorch_models", setting_name + '_' + verbose_qnet))
		print('Q net train finished --')

		print('plotting Qnet --')
		plot_envelope(Q_from_gt, states, actions, gts, \
					  setting_name + verbose_qnet,
					  seed, plot_func='q')

	elif qlearn_type == 'learn_border_data':

		print('Q net train starts ==')
		verbose_qnet = 'uebor%s_qgts%s'%(border, seed) if clip is None else 'uebor%s_clip%s_qgts%s'%(border, clip, seed)

		Q_from_gt = reg_qnet_to_batch(selected_states, selected_actions, selected_gts, state_dim, action_dim, device, seed,
									q_learning_rate=lr, q_weight_decay=wd, max_step_num=max_q_trainsteps, k=1)

		torch.save(Q_from_gt.state_dict(), '%s/%s_Qgt.pth' % ("./pytorch_models", setting_name+'_'+verbose_qnet))
		print('Q net train finished --')

	else: raise ValueError



def select_batch_ue_border(replay_buffer, setting_name, vue_sup, buffer_name, state_dim, border, clip):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_name, allow_pickle=True)
	actions = np.load('./results/ueMC_%s_A.npy' % buffer_name, allow_pickle=True)
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)

	upper_envelope = Value(state_dim, activation='relu')
	k_list = [10000, 1000, 100]
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name+vue_sup)))


	# compute a ceiling for better estimation of ue
	C = get_ue_ceiling(upper_envelope, states, clip)

	'''begin selection'''
	print('Selecting with ue border', border)
	selected_batch = utils.SARSAReplayBuffer()
	selected_states, selected_actions, selected_gts = [], [], []
	print(states.shape)
	for i in range(states.shape[0]):
		s, a, gt = states[i], actions[i], gts[i]
		s_val = upper_envelope(torch.from_numpy(s).unsqueeze(dim=0).float())
		s_val = torch.min(s_val, C)
		if gt >= border * s_val.detach().numpy():
			data = replay_buffer.index(i)
			selected_batch.add(data)
			selected_states.append(s)
			selected_actions.append(a)
			selected_gts.append(gt)

	initial_len, selected_len = replay_buffer.get_length(), selected_batch.get_length()
	print(selected_len, '/', initial_len, 'selecting ratio:', selected_len/initial_len)

	selection_info = 'ue_border%s' % border
	selection_info += '_clip%s' % clip if clip is not None else ''
	selected_batch.save( selection_info +'_'+ buffer_name)
	np.save('./results/sele%s_ueMC_%s_Gt' % (selection_info, setting_name), selected_gts)

	return (selected_states, selected_actions, selected_gts)

def get_ue_ceiling(upper_envelope, states, clip):
	states = torch.from_numpy(np.array(states))
	upper_envelope_r = []
	for i in range(states.shape[0]):
		s = states[i]
		upper_envelope_r.append(upper_envelope(s.float()).detach())

	upper_envelope_r = torch.stack(upper_envelope_r)
	increasing_ue_returns, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))

	if clip is not None:
		M = int(states.shape[0] * clip) - 1
		C = increasing_ue_returns[0, M]
		print("The clipping value is: ", C)
	else:
		C = increasing_ue_returns[0, -1]
		print("The highest ue : ", C, "No clipping")

	return C


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="MedSACTest1000")  # Prepends name to filename.
	parser.add_argument("--buffer_size", default="1000K")
	parser.add_argument("--cut_buffer_size", default="1000K")
	parser.add_argument("--buffer_seed", default=0, type=int)
	parser.add_argument("--qlearn_type", default="learn_all_data")
	parser.add_argument("--qloss_k", default=1, type=int)
	parser.add_argument("--border", default=0.75, type=float)
	parser.add_argument("--clip", default=None, type=float)
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs('MC_Q', args.seed)

	batch_qnet_train(env_set=args.env_set, seed=args.seed, buffer_seed=args.buffer_seed,
			 		 buffer_type=args.buffer_type, buffer_size=args.buffer_size, cut_buffer_size=args.cut_buffer_size,
					 qlearn_type=args.qlearn_type, qloss_k=args.qloss_k, border=args.border, clip=args.clip)

