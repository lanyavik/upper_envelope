import argparse
import gym
import os
import sys
import pickle
import time
import math
import torch
_dirpath =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/ue'
print('Directory:', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(_dirpath)

from utils import *
from models.mlp_critic import Value
from torch import LongTensor
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
#from core.agent import Agent
#from utils.replay_memory import Memory

import matplotlib.pyplot as plt



def update_params(batch, num_steps, optim_epochs):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    lr_mult = max(1.0 - float(num_steps) / args.max_step_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def calculate_returns(batch,expert_data):
    tau = 1
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    next_states = torch.from_numpy(np.stack(batch.next_state))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states)).data
    
    perm = np.arange(states.shape[0])
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, tau, use_gpu)
    num = 0
    for i in perm:
        expert_data.push(states[i], actions[i], masks[i], next_states[i], returns[i])
        num += 1


def L2PenaltyLoss(predicted,target,k_val):
    perm = np.arange(predicted.shape[0])
    loss = Variable(torch.Tensor([0]),requires_grad=True)
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

'''Training code for UE is here'''

def train_upper_envelope(states, actions, returns, state_dim, device, seed,
                         upper_learning_rate=3e-3,
                         weight_decay = 0.02,
                         max_step_num = int(1e6),
                         consecutive_steps = 4, k=10000):

    states = torch.from_numpy(np.array(states))
    actions = torch.from_numpy(np.array(actions))
    returns = torch.from_numpy(np.array(returns))  # reward is actually returns

    use_gpu = True if device == "cuda:0" else False

    # Init upper_envelope net (*use relu as activation function
    upper_envelope = Value(state_dim, activation='relu')
    upper_envelope_retrain = Value(state_dim, activation='relu')
    optimizer_upper = torch.optim.Adam(upper_envelope.parameters(), lr=upper_learning_rate,
                                       weight_decay=weight_decay)
    optimizer_upper_retrain = torch.optim.Adam(upper_envelope_retrain.parameters(), lr=upper_learning_rate,
                                               weight_decay=weight_decay)

    if use_gpu:
        upper_envelope = upper_envelope.cuda()
        upper_envelope_retrain = upper_envelope_retrain.cuda()


    # =========================== #
    # Split data into training and testing #
    # But make sure the highest Ri is in the training set

    # pick out the highest data point
    highestR, indice = torch.max(returns, 0)
    highestR = highestR.view(-1, 1)
    highestS = states[indice]
    highestA = actions[indice]
    print ("HighestR:",highestR)

    statesW = torch.cat((states[:indice],states[indice+1:]))
    actionsW = torch.cat((actions[:indice],actions[indice+1:]))
    returnsW = torch.cat((returns[:indice],returns[indice+1:]))

    # shuffle the data
    perm = np.arange(statesW.shape[0])
    np.random.shuffle(perm)
    perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)
    statesW, actionsW, returnsW = statesW[perm], actionsW[perm], returnsW[perm]

    # divide data into train/test
    divide = int(states.shape[0]*0.8)
    train_states, train_actions, train_returns = statesW[:divide], actionsW[:divide], returnsW[:divide]
    test_states, test_actions, test_returns = statesW[divide:], actionsW[divide:], returnsW[divide:]

    # add the highest data into training
    print(train_states.size(), highestS.size())
    print(train_actions.size(), highestA.size())
    print (train_returns.size(), highestR.size())
    train_states = torch.cat((train_states.squeeze(), highestS.unsqueeze(0)))
    train_actions = torch.cat((train_actions.squeeze(), highestA.unsqueeze(0)))
    train_returns = torch.cat((train_returns.squeeze(), highestR.squeeze().unsqueeze(0)))

    # train upper envelope
    # env_dummy = env_factory(0)
    # state_dim = env_dummy.observation_space.shape[0]
    # upper_envelope = Value(state_dim)
    # optimizer = torch.optim.Adam(upper_envelope.parameters(), lr=0.003, weight_decay=20)

    epoch_n = 100
    batch_size = 64
    optim_iter_num = int(math.ceil(train_states.shape[0] / batch_size))


    num_increase = 0
    previous_loss = math.inf

    calculate_vali = 2
    best_parameters = upper_envelope.state_dict()
    running_traning_steps = 0
    best_training_steps = running_traning_steps


    # Upper Envelope Training starts
    upper_envelope.train()

    while num_increase < consecutive_steps:
        # update theta for n steps, n = calculate_vali
        # train calculate_vali steps
        for i in range(calculate_vali):
            train_loss = 0
            perm = np.arange(train_states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

            train_states, train_actions, train_returns = train_states[perm], train_actions[perm], train_returns[perm]

            for i in range(optim_iter_num):
                ind = slice(i * batch_size, min((i + 1) * batch_size, states.shape[0]))
                states_b, returns_b = train_states[ind], train_returns[ind]
                states_b = Variable(states_b.float())
                returns_b = Variable(returns_b.float())
                Vsi = upper_envelope(states_b)
                # loss = loss_fn(Vsi, returns_b)
                loss = L2PenaltyLoss(Vsi, returns_b, k_val=k)
                train_loss += loss
                upper_envelope.zero_grad()
                loss.backward()
                optimizer_upper.step()

        # early stopping

        running_traning_steps += calculate_vali

        # calculate validation error
        test_iter = int(math.ceil(test_states.shape[0] / batch_size))
        validation_loss = 0
        for n in range(test_iter): 
            ind = slice(n * batch_size, min((n + 1) * batch_size, states.shape[0]))
            states_t, returns_t = test_states[ind], test_returns[ind]
            states_t = Variable(states_t.float())
            returns_t = Variable(returns_t.float())
            Vsi = upper_envelope(states_t)
            loss = L2PenaltyLoss(Vsi, returns_t, k_val=k)
            validation_loss += loss

        if validation_loss < previous_loss:
            best_training_steps = running_traning_steps
            previous_loss = validation_loss
            best_parameters = upper_envelope.state_dict()
            num_increase = 0
        else:
            num_increase += 1

    print ("best_training_steps:", best_training_steps)
    upper_envelope.load_state_dict(best_parameters)


    # retrain on the whole set
    upper_envelope_retrain.train()

    optim_iter_num = int(math.ceil(states.shape[0] / batch_size))
    for i in range(best_training_steps):
        #train_loss = 0
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)

        states, actions, returns = states[perm], actions[perm], returns[perm]

        for i in range(optim_iter_num):
            ind = slice(i * batch_size, min((i + 1) * batch_size, states.shape[0]))
            states_b, returns_b = states[ind], returns[ind]
            states_b = Variable(states_b.float())
            returns_b = Variable(returns_b.float())
            Vsi = upper_envelope_retrain(states_b)
            #loss = loss_fn(Vsi, returns_b)
            loss = L2PenaltyLoss(Vsi, returns_b, k_val=k)
            train_loss += loss
            upper_envelope_retrain.zero_grad()
            loss.backward()
            optimizer_upper_retrain.step()

    upper_envelope.load_state_dict(upper_envelope_retrain.state_dict())
    print("Policy training is complete.")

    return upper_envelope

'''Plotting code for UE is here'''

def plot_envelope(upper_envelope, states, actions, returns, buffer_name, seed):


    upper_learning_rate, weight_decay, max_step_num ,consecutive_steps = 3e-3, 0.02, int(1e6), 4

    states = torch.from_numpy(np.array(states))
    actions = torch.from_numpy(np.array(actions))
    returns = torch.from_numpy(np.array(returns))  # reward is actually returns
    highestR, indice = torch.max(returns, 0)
    highestR = highestR.view(-1, 1)
    highestS = states[indice]
    highestA = actions[indice]

    print("Plotting results...")
    '''
    plot_x_1 = []
    plot_y_1 = []

    plot_y_2 = []

    plot_y_1.append(train_loss.item()/train_states.shape[0])
    plot_y_2.append(validation_loss.item()/test_states.shape[0])

    plt.plot(plot_x_1, plot_y_1, label='Training loss')
    plt.plot(plot_x_1, plot_y_2, label='Validation loss')
    plt.legend()
    plt.xlabel('num_epoch')
    plt.grid()
    plt.title(buffer_name)
    plt.savefig('./plots/' + "ue_loss.png")
    '''

    # ======================= #
    # Sanity Check 
    # ======================= #

    # a) If all or almost all of the data points (test and validation) are below your envelope V(s)
    perm = np.arange(states.shape[0])
    num_above = 0
    for s in perm:
        state = states[s]
        re = returns[s]
        Vs = upper_envelope(state.float()).detach()
        if Vs < re.float():
            num_above += 1
    print ("number_above:", num_above)

    # b) V(si) approx equals Ri, where Ri is highest overall return
    Vs_highest = upper_envelope(highestS.float()).detach()
    print ("upper envelope Ri:", Vs_highest)
    print ("highest Ri:", highestR)


    upper_envelope_r = []
    MC_r = []
    for i in range(states.shape[0]):
        s = states[i]
        upper_envelope_r.append(upper_envelope(s.float()).detach())
        MC_r.append(returns[i])

    upper_envelope_r = torch.stack(upper_envelope_r)
    MC_r = torch.stack(MC_r)

    increasing_ue_returns, increasing_ue_indices = torch.sort(upper_envelope_r.view(1, -1))
    MC_r = MC_r[increasing_ue_indices[0]]

    plot_s = list(np.arange(states.shape[0]))
    plt.scatter(plot_s, list(MC_r.view(1, -1).numpy()[0]), s=0.5, color='orange', label='MC_Returns')
    plt.plot(plot_s, list(increasing_ue_returns.view(1, -1).numpy()[0]), label="UpperEnvelope")
    title = buffer_name +'_enve_vs_mc_maxsteps_' + str(max_step_num) + '_ulr_' + str(upper_learning_rate) \
            + '_wd_' + str(weight_decay) + '_seed_' + str(seed) + '_con_steps_' + str(consecutive_steps)

    plt.xlabel('state')
    plt.ylabel('V(s) comparison')
    plt.title(buffer_name+\
              '\n__mc_avg=%.2f'%MC_r.mean().item()+\
              '\n__above=%s_highUE=%.2f_highMC=%.2f'%(num_above, Vs_highest.item(), highestR.item()) )
    plt.legend()
    plt.savefig('./plots/' + "ue_visualization_%s.png"%title)
    #plt.savefig('/gpfsnyu/home/yw1370/PPO_Rejection/PyTorch-RL/images/UpperEnvelope' + title + '.png')
    plt.close('all')

    print('Plotting finished')


   




