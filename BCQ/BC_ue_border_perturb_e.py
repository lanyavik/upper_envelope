import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from spinup.algos.BCQ import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


class BC_ue_perturb(object):
	def __init__(self, state_dim, action_dim, max_action, lr, wd, Q_from_gt):

		self.Q_net = Q_from_gt.to(device)

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train_a_tilda(self, replay_buffer, max_updates=10, search_lr=1e-2, epsilon=torch.FloatTensor([0])):
		epsilon = epsilon.squeeze().to(device)
		data_length = replay_buffer.get_length()
		print('data_length:', data_length)
		state_list, perturbed_action_list = [], []
		update_num = []
		for ind in range(data_length):
			state, _, action, _, _, _ = replay_buffer.index(ind)

			state = torch.FloatTensor(state).unsqueeze(0).to(device)
			action = torch.FloatTensor(action).unsqueeze(0).to(device)
			#print(state.size(),action.size(),next_state.size(),next_action.size(), reward.size(),done.size())
			#print('original action', action)

			original_action = action.clone().detach()
			trn_action = action.requires_grad_()
			action_search_optimizer = torch.optim.Adam([trn_action], lr=search_lr)

			# Maximize the Q function w.r.t actor, subject to |critic-actor| < epsilon
			for update in range(max_updates):
				action_before_update = action.detach()
				loss = - self.Q_net(torch.cat([state, trn_action], 1))
				#print(action_before_update)
				#print(loss.item())
				
				action_search_optimizer.zero_grad()
				loss.backward()
				action_search_optimizer.step()
				
				constraint = torch.norm(trn_action.detach() - original_action, 2) - epsilon   # size []
				#print(eps.size(), constraint.size())
				#print(constraint.item())
				if constraint.detach() > 0: break
				
			state_list.append(state.detach())
			perturbed_action_list.append(action_before_update)
			update_num.append(update+1)
			#print(update, perturbed_action.detach(), constraint, eps)
				
		self.state_list = state_list
		self.perturbed_action_list = perturbed_action_list
		update_num = np.array(update_num)
		print('update number mean', np.mean(update_num), 'std', np.std(update_num), \
			  'max', np.max(update_num), 'min', np.min(update_num))


	def behavioral_cloning(self, iterations=500, batch_size=100, logger=dict()):
		
		for it in range(iterations):
			idxs = np.random.randint(0, len(self.state_list), size=batch_size).astype(int)
		
			state, action = torch.stack(self.state_list)[idxs], torch.stack(self.perturbed_action_list)[idxs]
			state = state.squeeze().to(device)
			action = action.squeeze().to(device)
			#print(state.shape, action.shape)
			
			# Compute MSE loss
			actor_loss = (self.actor(state) - action).pow(2).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(BCLoss=actor_loss.cpu().item())

