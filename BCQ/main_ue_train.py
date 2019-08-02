import gym
import numpy as np
import torch
import argparse
import os
from spinup.algos.BCQ import utils
from spinup.algos.ue.PPO_UE import train_upper_envelope, plot_envelope

from spinup.algos.ue.models.mlp_critic import Value


def ue_train(env_set="Hopper-v2", seed=1, buffer_type="FinalSigma0.5", buffer_seed=1, buffer_size='100K',
				max_ue_trainsteps=1e6, logger_kwargs=dict()):

	rollout_list = [None, 1000, 200, 100, 10]
	k_list = [10000, 1000, 100]
	print('testing MClength:', rollout_list[seed%10])
	print('Training loss ratio k:', k_list[seed//10])

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)


	buffer_name = "%s_%s_%s_%s" % (buffer_type, env_set, buffer_seed, buffer_size)
	setting_name = "%s_%s" % (buffer_name, seed)
	print("---------------------------------------")
	print("Settings: " + setting_name)
	print("---------------------------------------")

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
	replay_buffer.load(buffer_name)
	print('buffer name:', buffer_name)

	# extract (s,a,r) pairs from replay buffer
	length = replay_buffer.get_length()
	print(length)
	states, actions, gts = [], [], []
	for ind in range(length):
		state, _, action, _, _, dint = replay_buffer.index(ind)
		gt =  calculate_mc_ret(replay_buffer, ind, rollout=rollout_list[seed%10])
		states.append(state)
		actions.append(action)
		gts.append(gt)

	np.save('./results/ueMC_%s_S' % setting_name, states)
	np.save('./results/ueMC_%s_A' % setting_name, actions)
	np.save('./results/ueMC_%s_Gt' % setting_name, gts)

	print('ue train starts ==')

	states = np.load('./results/ueMC_%s_S.npy' % setting_name, allow_pickle=True)
	actions = np.load('./results/ueMC_%s_A.npy' % setting_name, allow_pickle=True)
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)

	upper_envelope = train_upper_envelope(states, actions, gts, state_dim, device, seed, k=k_list[seed//10])
	torch.save(upper_envelope.state_dict(), '%s/%s_UE.pth' % ("./pytorch_models", setting_name))
	print('ue train finished --')

	print('plotting ue --')

	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name)))

	plot_envelope(upper_envelope, states, actions, gts, \
				  setting_name+'k=%s_MClen=%s_gamma=%s'%(k_list[seed//10], rollout_list[seed%10], 0.99), seed)



def calculate_mc_ret(replay_buffer, idx, rollout=1000, discount=0.99):
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
	args = parser.parse_args()



	ue_train(env_set=args.env_set, seed=args.seed,
			 buffer_type=args.buffer_type, buffer_size=args.buffer_size)

