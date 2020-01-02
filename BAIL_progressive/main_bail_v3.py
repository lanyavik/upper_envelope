import gym
import numpy as np
import torch
import argparse
import os

from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.BAIL_progressive import utils, algo_BAIL

# check directory
if os.getcwd().find('lanya') == -1:
	os.chdir("/gpfsnyu/scratch/xc1305")
print('data directory', os.getcwd())
# check pytorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on device:", device)

def bail_learn(env_set="Hopper-v2", seed=0, buffer_type='FinalSigma0.5_env_0_1000K',
				gamma=0.99, ue_rollout=1000, augment_mc=True,
				C=None,
			    eval_freq=5000, max_timesteps=int(1e6), batch_size=1000,
			    lr=1e-3, wd=0, ue_lr=3e-3, ue_wd=2e-2, ue_loss_k=10000, ue_vali_freq=100,
				pct_anneal_type='linear', last_pct=0.25,
			   	pct_info_dic={},
				select_type='border',
			    logger_kwargs=dict()):

	"""set up logger"""
	global logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	if not os.path.exists("./plots"):
		os.makedirs("./plots")
	if not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")


	file_name = "BAIL-v3_%s_%s" % (env_set, seed)
	setting_name = "%s_r%s_g%s" % (buffer_type.replace('env', env_set), ue_rollout, gamma)
	setting_name += '_noaug' if not (augment_mc) else ''
	setting_name += '_augNew' if augment_mc == 'new' else ''

	print("---------------------------------------")
	print("Algo: " + file_name +"\tData: " + buffer_type)
	print("Settings: " + setting_name)
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

	# Load buffer
	if 'sac' in buffer_type:
		replay_buffer = utils.BEAR_ReplayBuffer()
		desire_stop_dict = {'Hopper-v2': 1000, 'Walker2d-v2': 500, 'HalfCheetah-v2': 4000, 'Ant-v2': 750}
		buffer_name = buffer_type.replace('env', env_set).replace('crt', str(desire_stop_dict[env_set]))
		replay_buffer.load(buffer_name)
	elif 'FinalSigma' in buffer_type:
		replay_buffer = utils.ReplayBuffer()
		buffer_name = buffer_type.replace('env', env_set)
		replay_buffer.load(buffer_name)
	else:
		raise FileNotFoundError('! Unknown type of dataset %s'%buffer_type)

	# Load data for training UE
	states = np.load('./results/ueMC_%s_S.npy' % buffer_name, allow_pickle=True)
	gts = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	print('Load gts with gamma:', gamma, 'rollout length:', ue_rollout)

	# Start training
	print('-- Policy train starts --')
	# Initialize policy
	policy = algo_BAIL.BAIL(state_dim, action_dim, max_action, max_iters=max_timesteps, States=states, MCrets=gts,
							ue_lr=ue_lr, ue_wd=ue_wd,
							pct_anneal_type=pct_anneal_type, last_pct=last_pct, pct_info_dic=pct_info_dic,
							select_type=select_type, C=C)


	training_iters, epoch = 0, 0
	
	while training_iters < max_timesteps:
		epoch += 1
		ue = policy.train(replay_buffer, training_iters, iterations=eval_freq, batch_size=batch_size,
								ue_loss_k=ue_loss_k,  ue_vali_freq= ue_vali_freq,
								logger=logger)

		if training_iters >= max_timesteps - eval_freq:
			cur_ue_setting = '' + setting_name + '_lossk%s_s%s' % (ue_loss_k, seed)
			plot_envelope(ue, states, gts, cur_ue_setting, seed, [ue_lr, ue_wd, ue_loss_k, max_timesteps, 4])
			torch.save(ue.state_dict(), '%s/ue_net_%s.pth' % ("./pytorch_models", cur_ue_setting))

		avgtest_reward = evaluate_policy(policy, test_env)
		training_iters += eval_freq

		# log training info
		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('AverageTestEpRet', avgtest_reward)
		logger.log_tabular('TotalSteps', training_iters)
		logger.log_tabular('CloneLoss', average_only=True)
		logger.log_tabular('UELoss', average_only=True)
		logger.log_tabular('UEValiLossMin', average_only=True)
		logger.log_tabular('BatchUEtrnSize', average_only=True)
		logger.log_tabular('SVal', with_min_and_max=True)
		logger.log_tabular('SelePct', average_only=True)
		logger.log_tabular('BatchUpSize', with_min_and_max=True)
		if select_type == 'border':
			logger.log_tabular('Border', with_min_and_max=True)
		elif select_type == 'margin':
			logger.log_tabular('Margin', with_min_and_max=True)
		else:
			raise Exception('! undefined selection type')

		logger.dump_tabular()


def plot_envelope(upper_envelope, states, returns, setting, seed, hyper_lst, make_title=True):

	upper_learning_rate, weight_decay, k_val, max_step_num, consecutive_steps = hyper_lst

	states = torch.from_numpy(states).to(device)
	upper_envelope_r = []
	for i in range(states.shape[0]):
		s = states[i]
		upper_envelope_r.append(upper_envelope(s.float()).detach())
		#highestR, _ = torch.max(returns, 0)
	MC_r = torch.from_numpy(returns).float().to(device)

	upper_envelope_r = torch.stack(upper_envelope_r)
	increasing_ue_vals, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))
	MC_r = MC_r[increasing_ue_indices[0]]

	#clipped_increasing_ue_vals = torch.where(increasing_ue_vals > Adapt_Clip, Adapt_Clip, increasing_ue_vals)
	all_ue_loss = torch.nn.functional.relu(increasing_ue_vals-MC_r).sum() + \
				  torch.nn.functional.relu(MC_r-increasing_ue_vals).sum()*k_val

	import matplotlib.pyplot as plt
	plt.rc('font', size=18)  # controls default text sizes
	plt.rc('axes', titlesize=10)  # fontsize of the axes title
	plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
	plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
	plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
	plt.rc('legend', fontsize=14)  # legend fontsize

	plot_s = list(np.arange(states.shape[0]))
	plt.scatter(plot_s, list(MC_r.view(1, -1).cpu().numpy()[0]), s=0.5, color='orange', label='MC_Returns')
	plt.plot(plot_s, list(increasing_ue_vals.view(1, -1).cpu().numpy()[0]), color='blue', label="UpperEnvelope")

	ue_info = '_loss_%.2fe6' % (all_ue_loss.item()/1e6)
	if make_title:
		title = setting.replace('_r', '\nr') + ue_info
		plt.title(title)
	plt.xlabel('state')
	plt.ylabel('V(s) comparison')
	plt.legend()
	plt.tight_layout()
	plt.savefig('./plots/' + "v_ue_visual_%s.png" % setting)
	plt.close('all')

	print('Plotted current UE in', "v_ue_visual_%s.png" % setting)

	return all_ue_loss


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

"""
def get_ue_clipping_info(ue_seed, ue_loss_k, detect_interval, setting_name, state_dim, buffer_info, ue_setting):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_info, allow_pickle=True)
	returns = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(
		torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (ue_seed, ue_loss_k))))

	clipping_val, clipping_loss = plot_envelope_with_clipping(upper_envelope, states, returns, buffer_info+ue_setting, ue_seed,
								  hyper_default=True, S=detect_interval)

	return clipping_val, clipping_loss


def select_batch_ue(replay_buffer, setting_name, buffer_info, state_dim, best_ue_seed, ue_loss_k, C, select_percentage):

	states = np.load('./results/ueMC_%s_S.npy' % buffer_info, allow_pickle=True)
	returns = np.load('./results/ueMC_%s_Gt.npy' % setting_name, allow_pickle=True)
	upper_envelope = Value(state_dim, activation='relu')
	upper_envelope.load_state_dict(
		torch.load('%s/%s_UE.pth' % ("./pytorch_models", setting_name + '_s%s_lok%s' % (best_ue_seed, ue_loss_k))))


	ratios = []
	for i in range(states.shape[0]):
		s, gt = torch.FloatTensor([states[i]]), torch.FloatTensor([returns[i]])
		s_val = upper_envelope(s.unsqueeze(dim=0).float()).detach().squeeze()
		ratios.append(gt / torch.min(s_val, C) if C is not None else gt / s_val)
	ratios = torch.stack(ratios).view(-1)
	increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
	bor_ind = increasing_ratio_indices[-int(select_percentage*states.shape[0])]
	border = ratios[bor_ind]

	'''begin selection'''
	selected_buffer = utils.ReplayBuffer()
	print('Selecting with ue border', border.item())
	for i in range(states.shape[0]):
		rat = ratios[i]
		if rat >= border:
			data = replay_buffer.index(i)
			selected_buffer.add(data)

	initial_len, selected_len = replay_buffer.get_length(), selected_buffer.get_length()
	print(selected_len, '/', initial_len, 'selecting ratio:', selected_len/initial_len)

	selection_info = 'ue_C%.2f' % C if C is not None else 'ue_none'
	selection_info += '_bor%.2f_len%s' % (border, selected_len)
	selected_buffer.save(selection_info +'_'+ buffer_info)

	return (selected_buffer, selected_len, border)
	"""



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_set", default="Hopper-v2")				# OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=int(1e2), type=int)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=int(3e2), type=int)		# Max time steps to run environment for
	parser.add_argument('--exp_name', type=str, default='bailv3_local')
	args = parser.parse_args()

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	bail_learn(env_set=args.env_set, seed=args.seed,
                eval_freq=args.eval_freq,
                max_timesteps=args.max_timesteps,
                logger_kwargs=logger_kwargs)
