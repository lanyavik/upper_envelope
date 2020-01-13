import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL_progressive import utils

if os.getcwd().find('lanya') == -1:
	os.chdir("/gpfsnyu/scratch/xc1305")
print('data directory', os.getcwd())

def get_mc(env_set="Hopper-v2", seed=1, buffer_type="sacpolicy_env_stopcrt_2_det_bear", cut_buffer_size='1000K',
           gamma=0.99, rollout=1000, augment_mc=True,
		   logger_kwargs=dict()):

	print('MClength:', rollout)
	print('Discount value', gamma)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("running on device:", device)

	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	if not os.path.exists("./results"):
		os.makedirs("./results")

	setting_name = "%s_r%s_g%s" % (buffer_type.replace('env', env_set), rollout, gamma)
	setting_name += 'noaug' if not (augment_mc) else ''
	print("---------------------------------------")
	print("Settings: " + setting_name)
	print("---------------------------------------")

	# Load buffer
	if 'sac' in buffer_type:
		replay_buffer = utils.BEAR_ReplayBuffer()
		desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
		buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
		replay_buffer.load(buffer_name)
		buffer_name += '_1000K'
		setting_name = setting_name.replace('crt', str(desire_stop_dict[env_set]))
	elif 'FinalSigma' in buffer_type:
		replay_buffer = utils.ReplayBuffer()
		buffer_name = buffer_type.replace('env', env_set)
		replay_buffer.load(buffer_name)
	else:
		raise FileNotFoundError('! Unknown type of dataset %s' % buffer_type)


	print('Starting MC calculation, type:', augment_mc)

	if augment_mc == 'gain':

		states, gains = calculate_mc_gain(replay_buffer, rollout=rollout, gamma=gamma)
		if not os.path.exists('./results/ueMC_%s_S.npy' % buffer_name):
			np.save('./results/ueMC_%s_S' % (buffer_name + '_' + cut_buffer_size), states)
		np.save('./results/ueMC_%s_Gain' % setting_name, gains)
	else:
            raise Exception('! undefined mc calculation type')

	print('Calculation finished ==')


def calculate_mc_gain(replay_buffer, rollout=1000, gamma=0.99):

    gts = []
    states = []
    actions = []

    g = 0
    prev_s = 0
    termination_point = 0

    endpoint = []
    dist = []  #L2 distance between the current state and the termination point

    length = replay_buffer.get_length()
    ep_len = 0

    #Find gains without augmentation term
    for ind in range(length-1, -1, -1):
        state, o2, action, r, done = replay_buffer.index(ind)

        states.append(state)
        actions.append(action)

        if done:
            g = r
            gts.append(g)
            endpoint.append(ind)
            termination_point = state
            prev_s = state
            dist.append(0)
            ep_len = 1
            continue

        if np.array_equal(prev_s, o2):
            g = gamma*g + r
            prev_s = state
            dist.append(np.linalg.norm(state - termination_point))
            ep_len += 1
        else:
            g = r
            endpoint.append(ind)
            termination_point = state
            prev_s = state
            dist.append(0)
            ep_len = 1

        gts.append(g)

    states = states[::-1]
    actions = actions[::-1]
    gts = gts[::-1]
    endpoint = endpoint[::-1]
    dist = dist[::-1]

    aug_gts = gts[:]

    #Add augmentation terms
    start = 0
    for i in range(len(endpoint)):
        end = endpoint[i]
        if end - start < rollout-1:
            #Early terminated episodes
            start = end+1
            continue

        #episodes not early terminated
        for j in range(end, start-1, -1):
            interval = dist[start: start + end-j+1]
            index = interval.index(min(interval))
            aug_gts[j] += gamma**(end-j+1) * gts[start+index]
            # print("Before aug: ", gts[j])
            # print("Discount: ", gamma**(rollout - (end-j+1)))
            # print("After aug: ", aug_gts[j])

        start = end+1

    return states, aug_gts


'''
def calculate_mc_ret(replay_buffer, idx, rollout=1000, discount=0.99):
	r_length = len(replay_buffer.storage['terminals'])
	r, d = replay_buffer.storage['rewards'][idx], replay_buffer.storage['terminals'][idx]
	mc_ret_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			pass
		else:
			r, d = replay_buffer.storage['rewards'][idx + h], replay_buffer.storage['terminals'][idx + h]
			mc_ret_est += discount ** h * r

	return np.asarray(mc_ret_est)


def calculate_mc_ret_truncate(replay_buffer, idx, rollout=1000, discount=0.99):
	r_length = replay_buffer.get_length()
	state, next_state, _, r, d = replay_buffer.index(idx)
	sampled_policy_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			break
		else:
			state, _, _, r, d = replay_buffer.index(idx + h)
			if (state == next_state).all():
				sampled_policy_est += discount ** h * r
				next_state = replay_buffer.index(idx + h)[1]
			else:
				break

	return np.asarray(sampled_policy_est)
'''


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds

	parser.add_argument("--gamma", default=0.99, type=float)
	parser.add_argument("--rollout", default=1000, type=int)
	args = parser.parse_args()

	exp_name = 'placeholder_mclen%s_gamma%s' % (args.rollout, args.gamma)
	logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

	get_mc(env_set=args.env_set, seed=args.seed,
			 gamma=args.gamma, rollout=args.rollout)

