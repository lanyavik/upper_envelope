import gym
import numpy as np
import torch
import argparse
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BCQ import utils, BC_ue_border_clip


from spinup.algos.ue.models.mlp_critic import Value


def bc_ue_learn(env_set="Hopper-v2", seed=0, buffer_type="FinalSigma0.5", buffer_seed=0, buffer_size='1000K',
                cut_buffer_size='1000K', ue_seed=2, gamma=0.99, ue_rollout=10, ue_loss_k=1000,
			    eval_freq=float(500), max_timesteps=float(1e5), lr=1e-3, wd=0, border=0.75, clip=None,
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
	replay_buffer = utils.SARSAReplayBuffer()
	replay_buffer.load(buffer_name + '_' + buffer_size)
	if buffer_size != cut_buffer_size:
		replay_buffer.cut_final(int(cut_buffer_size[:-1]) * 1e3)
	print(replay_buffer.get_length())

	print('buffer setting:', buffer_name + '_' + cut_buffer_size)

	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(ue_seed, ue_loss_k))))
	print('load envelope from', '%s/%s_UE.pth' % ("./pytorch_models", setting_name+'_s%s_lok%s'%(ue_seed, ue_loss_k)))
	print('testing MClength:', ue_rollout)
	print('Training loss ratio k:', ue_loss_k)
	K = ue_loss_k
	#plot_envelope(upper_envelope, states, actions, gts, buffer_name, seed)

	print('policy train starts --')

	states = np.load('./results/ueMC_%s_S.npy' % (buffer_name + '_' + cut_buffer_size), allow_pickle=True)
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)

	if clip is not None:
		upper_envelope_r = []

		states = torch.from_numpy(np.array(states))

		print(states.shape[0])

		for i in range(states.shape[0]):
			s = states[i]
			upper_envelope_r.append(upper_envelope(s.float()).detach())

		upper_envelope_r = torch.stack(upper_envelope_r)

		increasing_ue_returns, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))

		M = int(states.shape[0]*clip) - 1
		C = increasing_ue_returns[0, M].numpy()
	else:
		C = None

	print("The clipping value is: ", C)



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

def calculate_mc_ret(replay_buffer, idx, rollout=1000, discount=0.99):
	r_length = replay_buffer.get_length()
	state, next_state, a, _, r, d = replay_buffer.index(idx)
	sampled_policy_est = r
	for h in range(1, min(rollout, r_length-idx)):
		if bool(d):  # done=True if d=1
			break
		else:
			state, _, _, _, r, d = replay_buffer.index(idx + h)
			if (state == next_state).all():
				sampled_policy_est += discount ** h * r
				next_state = replay_buffer.index(idx + h)[1]
			else:
				break

	return np.asarray(sampled_policy_est)



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="MedSACtest1000")				# Prepends name to filename.
	parser.add_argument("--eval_freq", default=1e2, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bc_ue_b')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bc_ue_learn(env_set=args.env_name, seed=args.seed, buffer_type=args.buffer_type,
                eval_freq=args.eval_freq,
                max_timesteps=args.max_timesteps,
                logger_kwargs=logger_kwargs)
