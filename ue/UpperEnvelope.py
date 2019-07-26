import math
import sys
import os
import torch.utils.data as data_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *



def L2PenaltyLoss(predicted,target):
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
            mseloss = 10000 * (yi - Vsi)**2
            num += 1
        loss = torch.add(loss,mseloss) # a very big number
    #print ('below:',num)
    return loss/predicted.shape[0]


def best_training_epochs(upper_envelope, optimizer_upper, consecutive_steps, states, returns):
    # Split data into training and testing #
    # But make sure the highest Ri is in the training set

    # pick out the highest data point
    highestR, indice = torch.max(returns, 0)
    highestR = highestR.view(1, -1)
    highestS = states[indice].view(1,-1)
    print ("HighestR:",highestR)


    statesW = torch.cat((states[:indice],states[indice+1:]))
    returnsW = torch.cat((returns[:indice],returns[indice+1:]))

    # shuffle the data
    perm = np.arange(statesW.shape[0])
    np.random.shuffle(perm)
    perm = LongTensor(perm).cuda() if use_gpu else LongTensor(perm)
    statesW, returnsW = statesW[perm], returnsW[perm]

    # divide data into train/test
    divide = int(states.shape[0]*0.8)
    train_states, train_returns = statesW[:divide], returnsW[:divide]
    test_states, test_returns = statesW[divide:], returnsW[divide:]

    # add the highest data into training
    print (train_returns.shape)
    print (highestR.shape)
    train_states = torch.cat((train_states,highestS))
    train_returns = torch.cat((train_returns.view(-1,1),highestR))

    batch_size = 64
    train_dataset = data_utils.TensorDataset(train_states, train_returns)
    train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
    test_dataset = data_utils.TensorDataset(test_states, test_returns)
    test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=True)

    num_increase = 0
    previous_loss = math.inf
    calculate_vali = 2

    running_traning_steps = 0
    best_training_steps = running_traning_steps

    # Upper Envelope Training starts
    upper_envelope.train()
    traindata_size = train_states.shape[0]

    while num_increase < consecutive_steps:
        # update theta for n steps, n = calculate_vali
        # train calculate_vali steps
        for i in range(calculate_vali):
            epoch_train_loss = 0
            for i_batch, (X_batch, y_batch) in enumerate(train_loader):
                upper_envelope.zero_grad()

                Vsi = upper_envelope(X_batch)
                #loss = loss_fn(Vsi, returns_b)
                loss = L2PenaltyLoss(Vsi, y_batch)
                loss.backward()

                optimizer_upper.step()
                epoch_train_loss += loss.item()/traindata_size


        running_traning_steps += calculate_vali

        # calculate validation error
        validation_loss = 0
        for i_batch, (X_batch, y_batch) in enumerate(test_loader):
            Vsi = upper_envelope(X_batch)
            loss = L2PenaltyLoss(Vsi, y_batch)
            validation_loss += loss

        if validation_loss < previous_loss:
            best_training_steps = running_traning_steps
            previous_loss = validation_loss
            num_increase = 0
        else:
            num_increase += 1

    print ("best_training_steps:", best_training_steps)

    return best_training_steps, highestS




def train_upper_envelope(upper_envelope, optimizer_upper, best_training_steps, states, returns):
    upper_envelope.train()

    batch_size = 64
    train_dataset = data_utils.TensorDataset(states, returns)
    train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

    for i in range(best_training_steps):

        for i_batch, (X_batch, y_batch) in enumerate(train_loader):
            upper_envelope.zero_grad()

            Vsi = upper_envelope(X_batch)
            #loss = loss_fn(Vsi, returns_b)
            loss = L2PenaltyLoss(Vsi, y_batch)

            loss.backward()
            optimizer_upper.step()

    return upper_envelope


def value_net_step(returns, states, value_net, optimizer_value, l2_reg, optim_value_iternum = 1):
    values_target = Variable(returns)
    for _ in range(optim_value_iternum):
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()



