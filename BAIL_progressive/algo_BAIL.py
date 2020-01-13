import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# References
# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a)) 
		return a

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh', init_small_weights=False, init_w=1e-3):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if init_small_weights:
            for affine in self.affine_layers:
                affine.weight.data.uniform_(-init_w, init_w)
                affine.bias.data.uniform_(-init_w, init_w)


    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


class BAIL(object): # selection in mini-batch
	def __init__(self, state_dim, action_dim, max_action, max_iters, States, MCrets,
				 ue_lr=3e-3, ue_wd=2e-2, lr=1e-3, wd=0,
				 pct_anneal_type=None, last_pct=0.25, pct_info_dic=dict(),
				 select_type='border', C=None):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.v_ue = Value(state_dim, activation='relu').to(device)
		self.v_ue_optimizer = torch.optim.Adam(self.v_ue.parameters(), lr=ue_lr, weight_decay=ue_wd)
		self.best_v_ue = Value(state_dim, activation='relu').to(device)
		self.ue_best_parameters = self.v_ue.state_dict()

		self.MCrets = MCrets
		test_size = int(MCrets.shape[0] * 0.2)
		self.MC_valiset_indices = np.random.randint(0, MCrets.shape[0], size=test_size)

		self.test_states = torch.from_numpy(States[self.MC_valiset_indices])
		self.test_mcrets = torch.from_numpy(self.MCrets[self.MC_valiset_indices])
		print('ue test set size:', self.test_states.size(), self.test_mcrets.size())

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ue_valiloss_min = torch.Tensor([float('inf')]).to(device)
		self.num_increase = 0
		self.max_iters = max_iters
		self.pct_anneal_type = pct_anneal_type
		self.last_pct = last_pct
		self.pct_info_dic = pct_info_dic
		self.select_type = select_type
		self.C = C


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, done_training_iters, iterations=5000, batch_size=1000,
			  ue_loss_k=10000, ue_vali_freq=10,
			  logger=dict()):

		for it in range(done_training_iters, done_training_iters + iterations):

			# get batch data
			state, next_state, action, reward, done, idxs = replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			# next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			# done = torch.FloatTensor(1 - done).to(device)
			mc_ret = torch.FloatTensor(self.MCrets[idxs]).to(device)

			uetrain_batch_pos = [p for p, i in enumerate(idxs) if i not in self.MC_valiset_indices]
			uetrain_s = state[uetrain_batch_pos]
			uetrain_mc = mc_ret[uetrain_batch_pos]

			# train upper envelope by the k-penalty loss
			Vsi = self.v_ue(uetrain_s)
			ue_loss = L2PenaltyLoss(Vsi, uetrain_mc, k_val=ue_loss_k)

			self.v_ue_optimizer.zero_grad()
			ue_loss.backward()
			self.v_ue_optimizer.step()

			""" if it is time to recalculate border/margin """
			""" do validation for the UE network, update the best ue """
			if it % ue_vali_freq == 0:
				validation_loss = calc_ue_valiloss(self.v_ue, self.test_states, self.test_mcrets,
												   ue_bsize=int(batch_size * 0.8), ue_loss_k=ue_loss_k)

				# choose best parameters with least validation loss for the eval ue
				self.ue_valiloss_min = torch.min(self.ue_valiloss_min, validation_loss)
				logger.store(UEValiLossMin=self.ue_valiloss_min)
				if validation_loss > self.ue_valiloss_min:
					self.best_v_ue.load_state_dict(self.ue_best_parameters)
					self.num_increase += 1
				else:
					self.ue_best_parameters = self.v_ue.state_dict()
					self.num_increase = 0
				# if validation loss of ue is increasing for some consecutive steps, also return the training ue to least
				# validation loss parameters
				if self.num_increase == 4:
					self.v_ue.load_state_dict(self.ue_best_parameters)

			# estimate state values by the upper envelope
			state_value = self.best_v_ue(state).squeeze().detach()
			# project negative or small positive state values to (0, 1)
			state_value = torch.where(state_value < 1, (state_value - 1).exp(), state_value)
			if self.C is not None:
				C = self.C.to(device)
				state_value = torch.where(state_value > C, C, state_value)
			# print(type(state_value))

			# get current percentage
			if self.pct_anneal_type == 'constant':
				cur_pct = self.last_pct
			elif self.pct_anneal_type == 'linear':
				cur_pct = 1 - it / self.max_iters * (1 - self.last_pct)
			elif self.pct_anneal_type == 'line2const':
				const_timesteps = self.pct_info_dic['const_timesteps']
				cur_pct = 1 - min(it / (self.max_iters - const_timesteps), 1.0) * (1 - self.last_pct)
			elif self.pct_anneal_type == 'convex1': # sigmoid-shape function
				convex1_coef = self.pct_info_dic['convex1_coef']
				cur_pct = self.last_pct + 2*(1 - self.last_pct) / (1 + np.exp(convex1_coef * it / self.max_iters))
			else:
				raise Exception('! undefined percentage anneal type')

			logger.store(SelePct=cur_pct)

			# determine the border / margin by current percentage
			if self.select_type == 'border':
				ratios = mc_ret / state_value
				increasing_ratios, increasing_ratio_indices = torch.sort(ratios.view(-1))
				bor_ind = increasing_ratio_indices[-int(cur_pct * batch_size)]
				border = ratios[bor_ind]

				weights = torch.where(mc_ret >= border * state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
				logger.store(Border=border.cpu().item())

			elif self.select_type == 'margin':
				diffs = mc_ret - state_value
				increasing_diffs, increasing_diff_indices = torch.sort(diffs.view(-1))
				mrg_ind = increasing_diff_indices[-int(cur_pct * batch_size)]
				margin = diffs[mrg_ind]

				weights = torch.where(mc_ret >= margin + state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
				logger.store(Margin=margin.cpu().item())

			else:
				raise Exception('! undefined selection type')

			# Compute MSE loss for actor
			update_size = weights.sum().cpu().item()
			weights = torch.stack([weights, ] * self.action_dim, dim=1)
			# print(weights.size(), action.size())
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(CloneLoss=actor_loss.detach().cpu().item(), UELoss=ue_loss.detach().cpu().item(),
						 BatchUEtrnSize=len(uetrain_batch_pos), BatchUpSize=update_size,
						 SVal=state_value.detach().mean())

		return self.best_v_ue


class BAIL_selebuf(object):  # selection in the whole buffer
	def __init__(self, state_dim, action_dim, max_action, max_iters, States, MCrets,
				 ue_lr=3e-3, ue_wd=2e-2, lr=1e-3, wd=0,
				 pct_anneal_type=None, last_pct=0.25, pct_info_dic=dict(),
				 select_type='border', C=None):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.v_ue = Value(state_dim, activation='relu').to(device)
		self.v_ue_optimizer = torch.optim.Adam(self.v_ue.parameters(), lr=ue_lr, weight_decay=ue_wd)
		self.best_v_ue = Value(state_dim, activation='relu').to(device)
		self.ue_best_parameters = self.v_ue.state_dict()

		self.States = torch.from_numpy(States).float().to(device)
		self.MCrets = torch.from_numpy(MCrets).float().to(device)
		test_size = int(self.MCrets.shape[0] * 0.2)
		self.MC_valiset_indices = np.random.randint(0, MCrets.shape[0], size=test_size)
		print('ue test set size:', self.MC_valiset_indices.shape)

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ue_valiloss_min = torch.Tensor([float('inf')]).to(device)
		self.num_increase = 0
		self.max_iters = max_iters
		self.pct_anneal_type = pct_anneal_type
		self.last_pct = last_pct
		self.pct_info_dic = pct_info_dic
		self.select_type = select_type
		self.C = C

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, done_training_iters, iterations=5000, batch_size=1000, ue_loss_k=10000,
			  ue_vali_freq=1000,
			  logger=dict()):

		for it in range(done_training_iters, done_training_iters + iterations):

			# get batch data
			state, next_state, action, reward, done, idxs = replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			# next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			# done = torch.FloatTensor(1 - done).to(device)
			mc_ret = self.MCrets[idxs]
			uetrain_batch_pos = [p for p, i in enumerate(idxs) if i not in self.MC_valiset_indices]
			uetrain_s = state[uetrain_batch_pos]
			uetrain_mc = mc_ret[uetrain_batch_pos]

			# train upper envelope by the k-penalty loss
			Vsi = self.v_ue(uetrain_s)
			ue_loss = L2PenaltyLoss(Vsi, uetrain_mc, k_val=ue_loss_k)

			self.v_ue_optimizer.zero_grad()
			ue_loss.backward()
			self.v_ue_optimizer.step()

			""" if it is time to recalculate border/margin """
			""" 1. calculate validation loss from the validation set, update the best ue """
			if it % ue_vali_freq == 0:

				test_states, test_mcrets = self.States[self.MC_valiset_indices], self.MCrets[self.MC_valiset_indices]
				validation_loss = calc_ue_valiloss(self.v_ue, test_states, test_mcrets,
												   ue_bsize=int(batch_size * 0.8), ue_loss_k=ue_loss_k)

				# choose the best parameters with least validation loss for the eval ue
				self.ue_valiloss_min = torch.min(self.ue_valiloss_min, validation_loss)
				logger.store(UEValiLossMin=self.ue_valiloss_min)
				if validation_loss > self.ue_valiloss_min:
					self.best_v_ue.load_state_dict(self.ue_best_parameters)
					self.num_increase += 1
				else:
					self.ue_best_parameters = self.v_ue.state_dict()
					self.num_increase = 0
				# if validation loss of ue is increasing for some consecutive steps, also return the training ue to least
				# validation loss parameters
				if self.num_increase == 4:
					self.v_ue.load_state_dict(self.ue_best_parameters)

				""" 2. estimate state values by the beat ue """
				States_values = []
				num_slice = int(np.ceil(self.States.shape[0] / batch_size))
				for i in range(num_slice):
					ind = slice(i * batch_size, min((i + 1) * batch_size, self.States.shape[0]))
					s = self.States[ind]
					States_values.append(self.best_v_ue(s).detach())
				States_values = torch.stack(States_values).view(-1)
				# States_values = self.best_v_ue(self.States).squeeze().detach()

				# project negative or small positive state values to (0, 1)
				States_values = torch.where(States_values < 1, (States_values - 1).exp(), States_values)

				if self.C is not None:
					C = self.C.to(device)
					States_values = torch.where(States_values > C, C, States_values)

				""" 3. get current percentage from the anneal plan """
				if self.pct_anneal_type == 'constant':
					cur_pct = self.last_pct
				elif self.pct_anneal_type == 'linear':
					cur_pct = 1 - it / self.max_iters * (1 - self.last_pct)
				elif self.pct_anneal_type == 'line2const':
					const_timesteps = self.pct_info_dic['const_timesteps']
					cur_pct = 1 - min(it / (self.max_iters - const_timesteps), 1.0) * (1 - self.last_pct)
				elif self.pct_anneal_type == 'convex1':  # sigmoid-shape function
					convex1_coef = self.pct_info_dic['convex1_coef']
					cur_pct = self.last_pct + 2 * (1 - self.last_pct) / (1 + np.exp(convex1_coef * it / self.max_iters))
				else:
					raise Exception('! undefined percentage anneal type')

				logger.store(SelePct=cur_pct)

				""" 4.  determine the border / margin by current percentage """
				if self.select_type == 'border':
					ratios = self.MCrets.view(-1) / States_values
					increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
					bor_ind = increasing_ratio_indices[-int(cur_pct * self.States.shape[0])]
					border = ratios[bor_ind]
					logger.store(Border=border.cpu().item())
				# print(self.MCrets[increasing_ratio_indices[:100]], States_values[increasing_ratio_indices[:100]],
				#	   ratios[increasing_ratio_indices[:100]])

				elif self.select_type == 'margin':
					diffs = self.MCrets.view(-1) - States_values
					increasing_diffs, increasing_diff_indices = torch.sort(diffs)
					mrg_ind = increasing_diff_indices[-int(cur_pct * self.States.shape[0])]
					margin = diffs[mrg_ind]
					logger.store(Margin=margin.cpu().item())

				else:
					raise Exception('! undefined selection type')

			# Compute MSE loss for actor
			state_value = States_values[idxs].squeeze()
			# state_value_check = self.best_v_ue(state).squeeze().detach()
			# state_value_check = torch.where(state_value_check < 1, (state_value_check - 1).exp(), state_value_check)
			if self.select_type == 'border':
				weights = torch.where(mc_ret >= border * state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
			elif self.select_type == 'margin':
				weights = torch.where(mc_ret >= margin + state_value, \
									  torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))

			update_size = weights.sum().cpu().item()
			weights = torch.stack([weights, ] * self.action_dim, dim=1)
			# print(weights.size(), action.size())
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(CloneLoss=actor_loss.detach().cpu().item(), UELoss=ue_loss.detach().cpu().item(),
						 BatchUEtrnSize=len(uetrain_batch_pos), BatchUpSize=update_size,
						 SVal=state_value.detach().mean())

		return self.best_v_ue


def L2PenaltyLoss(predicted,target,k_val):
    perm = np.arange(predicted.shape[0])
    loss = Variable(torch.Tensor([0]),requires_grad=True).to(device)
    num = 0
    for i in perm:
        Vsi = predicted[i]
        yi = target[i]
        if Vsi >= yi:
            mseloss = (Vsi - yi)**2
            #loss = torch.add(loss,mseloss)
        else:
            mseloss = k_val * (yi - Vsi)**2
            num += 1
        loss = torch.add(loss, mseloss) # a very big number
    #print ('below:',num)
    return loss/predicted.shape[0]

def calc_ue_valiloss(upper_envelope, test_states, test_returns, ue_bsize, ue_loss_k):

	test_iter = int(np.ceil(test_returns.shape[0] / ue_bsize))
	validation_loss = torch.FloatTensor([0]).detach().to(device)
	for n in range(test_iter):
		ind = slice(n * ue_bsize, min((n + 1) * ue_bsize, test_returns.shape[0]))
		states_t, returns_t = test_states[ind], test_returns[ind]
		states_t = Variable(states_t.float()).to(device)
		returns_t = Variable(returns_t.float()).to(device)
		Vsi = upper_envelope(states_t)
		loss = L2PenaltyLoss(Vsi, returns_t, k_val=ue_loss_k).detach()
		validation_loss += loss

	return validation_loss


class BAIL_weight(object): # weighting by ratio in mini-batch
	def __init__(self, state_dim, action_dim, max_action, max_iters, States, MCrets,
				 ue_lr=3e-3, ue_wd=2e-2, lr=1e-3, wd=0,
				 C=None):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.v_ue = Value(state_dim, activation='relu').to(device)
		self.v_ue_optimizer = torch.optim.Adam(self.v_ue.parameters(), lr=ue_lr, weight_decay=ue_wd)
		self.best_v_ue = Value(state_dim, activation='relu').to(device)
		self.ue_best_parameters = self.v_ue.state_dict()

		self.MCrets = MCrets
		test_size = int(MCrets.shape[0] * 0.2)
		self.MC_valiset_indices = np.random.randint(0, MCrets.shape[0], size=test_size)

		self.test_states = torch.from_numpy(States[self.MC_valiset_indices])
		self.test_mcrets = torch.from_numpy(self.MCrets[self.MC_valiset_indices])
		print('ue test set size:', self.test_states.size(), self.test_mcrets.size())

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ue_valiloss_min = torch.Tensor([float('inf')]).to(device)
		self.num_increase = 0
		self.max_iters = max_iters
		self.C = C


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, done_training_iters, iterations=5000, batch_size=1000,
			  ue_loss_k=10000, ue_vali_freq=10,
			  logger=dict()):

		for it in range(done_training_iters, done_training_iters + iterations):

			# get batch data
			state, next_state, action, reward, done, idxs = replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			# next_state = torch.FloatTensor(next_state).to(device)
			# reward = torch.FloatTensor(reward).to(device)
			# done = torch.FloatTensor(1 - done).to(device)
			mc_ret = torch.FloatTensor(self.MCrets[idxs]).to(device)

			uetrain_batch_pos = [p for p, i in enumerate(idxs) if i not in self.MC_valiset_indices]
			uetrain_s = state[uetrain_batch_pos]
			uetrain_mc = mc_ret[uetrain_batch_pos]

			# train upper envelope by the k-penalty loss
			Vsi = self.v_ue(uetrain_s)
			ue_loss = L2PenaltyLoss(Vsi, uetrain_mc, k_val=ue_loss_k)

			self.v_ue_optimizer.zero_grad()
			ue_loss.backward()
			self.v_ue_optimizer.step()

			""" if it is time to recalculate border/margin """
			""" do validation for the UE network, update the best ue """
			if it % ue_vali_freq == 0:
				validation_loss = calc_ue_valiloss(self.v_ue, self.test_states, self.test_mcrets,
												   ue_bsize=int(batch_size * 0.8), ue_loss_k=ue_loss_k)

				# choose best parameters with least validation loss for the eval ue
				self.ue_valiloss_min = torch.min(self.ue_valiloss_min, validation_loss)
				logger.store(UEValiLossMin=self.ue_valiloss_min)
				if validation_loss > self.ue_valiloss_min:
					self.best_v_ue.load_state_dict(self.ue_best_parameters)
					self.num_increase += 1
				else:
					self.ue_best_parameters = self.v_ue.state_dict()
					self.num_increase = 0
				# if validation loss of ue is increasing for some consecutive steps, also return the training ue to least
				# validation loss parameters
				if self.num_increase == 4:
					self.v_ue.load_state_dict(self.ue_best_parameters)

			# estimate state values by the upper envelope
			state_value = self.best_v_ue(state).squeeze().detach()
			# project negative or small positive state values to (0, 1)
			state_value = torch.where(state_value < 1, (state_value - 1).exp(), state_value)
			if self.C is not None:
				C = self.C.to(device)
				state_value = torch.where(state_value > C, C, state_value)
			# print(type(state_value))

			# Compute MSE loss for actor
			weights = mc_ret / state_value
			update_size = weights.sum().cpu().item()
			weights = torch.stack([weights, ] * self.action_dim, dim=1)
			# print(weights.size(), action.size())
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(CloneLoss=actor_loss.detach().cpu().item(), UELoss=ue_loss.detach().cpu().item(),
						 BatchUEtrnSize=len(uetrain_batch_pos), BatchUpSize=update_size,
						 SVal=state_value.detach().mean())

		return self.best_v_ue