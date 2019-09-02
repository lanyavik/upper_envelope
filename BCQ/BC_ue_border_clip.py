import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from spinup.algos.BCQ import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



class BC_ue(object):
	def __init__(self, state_dim, action_dim, max_action, lr, wd, ue_valfunc, mc_rets):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)

		self.ue_valfunc = ue_valfunc
		self.mc_rets = mc_rets

		self.state_dim = state_dim
		self.action_dim = action_dim



	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations=500, batch_size=1000, discount=0.99, tau=0.005, border=0.95, C = None,
			  logger=dict()):

		for it in range(iterations):

			state, next_state, action, next_action, reward, done, idxs = \
				replay_buffer.sample(batch_size, require_idxs=True)

			state = torch.FloatTensor(state).to(device)
			action = torch.FloatTensor(action).to(device)
			next_state = torch.FloatTensor(next_state).to(device)
			next_action = torch.FloatTensor(next_action).to(device)
			reward = torch.FloatTensor(reward).to(device)
			done = torch.FloatTensor(1 - done).to(device)
			mc_ret = self.mc_rets[idxs]

			# estimate state values by upper envelope
			state_value = self.ue_valfunc(state.cpu()).squeeze().detach().numpy()


			if C is not None:
				state_value = np.where(state_value > C, C, state_value)

			weights = np.where(mc_ret >= border * state_value, 1, 0)
			update_size = np.count_nonzero(weights)
			weights = torch.FloatTensor(np.stack((weights,) * self.action_dim , axis=1)).to(device)
			#print(weights.size(), action.size())
			# Compute MSE loss
			actor_loss = torch.mul(weights, self.actor(state) - action).pow(2).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			logger.store(Loss=actor_loss.cpu().item(), UpSize=update_size, SVal=state_value.mean())

		#torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))



