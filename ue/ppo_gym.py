import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

#import matplotlib.pyplot as plt

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 2)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-step-num', type=int, default=int(1e6), metavar='N',
                    help='maximal number of steps (default: 1e6)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--m', type=int, default=1, metavar='N',
                    help="restart limit (default: 1)")
parser.add_argument('--divide-ratio', type=float, default=2, metavar='N',
                    help="restart limit (default: 2)")
parser.add_argument('--diff-ratio', type=float, default=2, metavar='N',
                    help="restart limit (default: 2)")
parser.add_argument('--diff-epochs', type=int, default=10, metavar='N',
                    help="restart limit (default: 10)")
args = parser.parse_args()


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)
    return env


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)

#state_dimension = the observation space
state_dim = env_dummy.observation_space.shape[0]

#is the action discrete or continuous
is_disc_action = len(env_dummy.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
    else:
        policy_net = Policy(state_dim, env_dummy.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
del env_dummy

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

"""create agent"""
agent = Agent(env_factory, policy_net, running_state=running_state, render=args.render, num_threads=args.num_threads, seed=args.seed, thread_id = 0)


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


def update_params_cut(batch, num_steps, cut):
    cut_theta = None
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

        if _ == cut:
            print ("store update at step "+str(_+1))
            cut_theta = policy_net.state_dict()
            

    return cut_theta

def main_loop():
    plot_rewards = []
    plot_k = []
    num_of_decrease = 0
    previous_reward = None
    num_episodes = []
    episode_rewards = []
    num_episode = 0
    num_steps = 0
    i_iter = 0
    while num_steps < args.max_step_num:
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        i_iter += 1
        #plot_k.append(num_episodes)
        if previous_reward == None:
            previous_reward = log['avg_reward']
        else:
            if previous_reward > log['avg_reward']:
                num_of_decrease += 1
            previous_reward = log['avg_reward']

        current_batch_episode_rewards = log['episode_rewards']
        num_episode += log['num_episodes']
        num_episodes.append(num_episode)
        episode_rewards += current_batch_episode_rewards
        plot_k.append(num_steps)
        plot_rewards.append(previous_reward)
        num_steps +=log['num_steps']

        t0 = time.time()
        update_params(batch, num_steps, 10)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()

        """clean up gpu memory"""
        
        torch.cuda.empty_cache()

    #plt.plot(plot_k, plot_rewards)
    #plt.xlabel('Num_Steps')
    #plt.ylabel('Average reward over the trajectories')

    title = args.env_name + '_vanilla_' + str(args.max_step_num) + '_' + str(args.seed) + '_no_annealing'
    #plt.title(title)
    #plt.savefig('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/images/'+ title + '.png')

    #with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ title +'.pickled', 'wb+') as f:
    #    pickle.dump([plot_k,plot_rewards], f)

    #with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ title +'_episode' +'.pickled', 'wb+') as f:
    #    pickle.dump(num_episodes, f)

    #with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ title +'_episode_rewards' +'.pickled', 'wb+') as f:
    #    pickle.dump(episode_rewards, f)

    # plt.plot(plot_k, plot_rewards)
    # plt.xlabel('Theta_updates')
    # plt.ylabel('Average reward over the trajectories')

    # title = args.env_name + '\nclip-epsilon: ' + str(args.clip_epsilon) + ' lr:' + str(args.learning_rate) \
    # + ' num_of_decrease: ' + str(num_of_decrease)
    # plt.title(title)
    # #plt.legend()
    # plt.savefig(title + '.png')

main_loop()
#theta = policy_net.state_dict()
#main_loop()

def restart():
    plot_rewards = []
    plot_k = []
    num_of_decrease = 0
    previous_reward = None
    theta_k = None
    cut_theta = None
    num_restart = 0
    m = 1
    i_iter = 0
    flag = True
    num_episodes = []
    num_episode = 0
    episode_rewards = []

    num_steps = 0
    total_restart = []
    temp = 0

    sta = []

    while num_steps < args.max_step_num:
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        if previous_reward == None:
            previous_reward = log['avg_reward']
            #save the first model parameters
            i_iter += 1
            #plot_k.append(num_episodes)
            plot_k.append(num_steps)
            plot_rewards.append(previous_reward)
            current_batch_episode_rewards = log['episode_rewards']
            episode_rewards += current_batch_episode_rewards

            current_batch_episode = log['num_episodes']
            num_episode += current_batch_episode
            num_episodes.append(num_episode)
            num_steps += log['num_steps']
            theta_k = policy_net.state_dict()
            flag = True
        else:
            #if J(theta_k) > J(theta _k+1):
            #       back to theta_k
            #       do not update parameters
            #       keep previous_reward unchanged
            #       increase num_restart by 1
            possible_thetas = [policy_net.state_dict()]
            average_rewards = [log['avg_reward']]
            cross_steps = [log['num_steps']]
            cross_episodes = [log['num_episodes']]
            temp_episodes = [current_batch_episode]
            temp_episode_rewards = [current_batch_episode_rewards]
            temp = [previous_reward]
            cross_batch_episode_rewards = [log['episode_rewards']]
            batches = [batch]
            while previous_reward - log['avg_reward'] > 0:
                policy_net.load_state_dict(theta_k)
                print("Restart, Reward: "+str(num_restart)+', '+str(previous_reward))

                num_restart += 1
                #env_dummy.seed(args.seed + num_restart)
                #generate new trajectories with old theta
                batch, log = agent.collect_samples(args.min_batch_size)
                previous_reward = log['avg_reward']
                temp.append(previous_reward)
                
                temp_episodes.append(log['num_episodes'])
                temp_episode_rewards.append(log['episode_rewards'])
                #num_episode += log["num_episodes"]
                num_steps += log['num_steps']
                
                #update parameters using new trajectories
                update_params(batch, i_iter, 10)
                #generate trajectories with new theta to see if its better
                batch, log = agent.collect_samples(args.min_batch_size)
                possible_thetas.append(policy_net.state_dict())
                average_rewards.append(log['avg_reward'])
                batches.append(batch)
                cross_steps.append(log['num_steps'])
                cross_episodes.append(log['num_episodes'])
                cross_batch_episode_rewards.append(log['episode_rewards'])

                if num_restart >= m:
                    total_restart.append(num_restart)
                    print("Restart, Reward: "+str(num_restart)+', '+str(previous_reward))
                    num_restart = 0
                    if previous_reward > log['avg_reward']:
                        flag = False
                        num_of_decrease += 1
                    break
            
            #   accept theta_k+1, save its parameters
            #   update previous_reward
            if not flag:
                previous_reward = max(average_rewards)
                index = average_rewards.index(previous_reward)

                diff = (temp[index] - max(average_rewards))/abs(temp[index])
                sta.append(diff)
                
                theta_k = possible_thetas[index]
                steps = cross_steps[index]
                current_batch_episode = cross_episodes[index]

                previous_episode = temp_episodes[index]
                previous_episode_reward = temp_episode_rewards[index]

                current_batch_episode_rewards = cross_batch_episode_rewards[index]
                other_steps = np.sum(cross_steps) - steps
                #other_epi = np.sum(cross_episodes) - epi
                batch = batches[index] 

                i_iter += 1
                #plot_k.append(num_episodes)
                num_steps += other_steps
                #num_episode += other_epi
                plot_k.append(num_steps)

                if len(num_episodes) > 1:
                    invalid_episode = num_episodes[-1] - num_episodes[-2]
                else:
                    invalid_episode = num_episodes[-1]

                num_episode = num_episode - invalid_episode + previous_episode
                num_episodes[-1] = num_episode

                num_episode += current_batch_episode 
                num_episodes.append(num_episode)

                invalid_episode_reward_length = len(temp_episode_rewards[0])
                episode_rewards = episode_rewards[:-invalid_episode_reward_length]
                episode_rewards += previous_episode_reward
                episode_rewards += current_batch_episode_rewards

                plot_rewards.append(previous_reward)
                num_steps += steps
                flag = True
            else:
                theta_k = policy_net.state_dict()
                #diff = (previous_reward - log['avg_reward'])/previous_reward
            
                previous_reward = log['avg_reward']
                i_iter += 1
                #plot_k.append(num_episodes)
                previous_episode = temp_episodes[-1]
                previous_episode_reward = temp_episode_rewards[-1]

                invalid_episode_reward_length = len(temp_episode_rewards[0])
                episode_rewards = episode_rewards[:-invalid_episode_reward_length]
                episode_rewards += previous_episode_reward

                current_batch_episode_rewards = log['episode_rewards']
                episode_rewards += current_batch_episode_rewards

                other_steps = np.sum(cross_steps) - log['num_steps']
                #other_epi = np.sum(cross_episodes) - log['num_episodes']
                num_steps += other_steps
                #num_episode += other_epi
                plot_k.append(num_steps)

                if len(num_episodes) > 1:
                    invalid_episode = num_episodes[-1] - num_episodes[-2]
                else:
                    invalid_episode = num_episodes[-1]

                num_episode = num_episode - invalid_episode + previous_episode
                num_episodes[-1] = num_episode

                current_batch_episode = log['num_episodes']
                num_episode += current_batch_episode
                num_episodes.append(num_episode)

                plot_rewards.append(previous_reward)
                num_steps += log['num_steps']

        
        t0 = time.time()
        cut_theta = update_params(batch, num_steps, 10)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], previous_reward))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    #with open(args.env_name+ '_vanilla'+'.pickled', 'rb') as f:
    #    object_file = list(pickle.load(f))
    #    plot_x_v = [x for x in object_file[0]]
    #    plot_r_v = [y for y in object_file[1]]

    #plt.plot(plot_k, plot_rewards)
    #plt.plot(plot_x_v, plot_r_v, label="vanilla")
    #plt.xlabel('Num_Steps')
    #plt.ylabel('Average reward over the trajectories')

    mean = np.mean(sta)
    std = np.std(sta)
    median = np.median(sta)

    title = args.env_name + '_restart_'+ str(args.max_step_num) + "_" + str(args.seed)
    #notes = str(np.sum(total_restart)) + '/' + str(np.sum(total_restart) + i_iter - len(total_restart)) + ' mean:'+str(round(mean, 6)) + " std:" + str(round(std, 6)) + " median:" + str(round(median, 6))
    #plt.title(title + '\n' + notes)
    #plt.legend()
    #plt.savefig('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/images/'+ title + '.png')

    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ title +'.pickled', 'wb+') as f:
        pickle.dump([plot_k,plot_rewards], f)

    statistics = args.env_name + "_" + str(args.max_step_num) + "_stat_" + str(args.seed) 
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ statistics +'.pickled', 'wb+') as f:
        pickle.dump(sta, f)

    ep = args.env_name + "_" + str(args.max_step_num) + "_restart_episode_" + str(args.seed) 
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ ep +'.pickled', 'wb+') as f:
        pickle.dump(num_episodes, f)

    epr = args.env_name + "_" + str(args.max_step_num) + "_restart_episode_rewards_" + str(args.seed) 
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ epr +'.pickled', 'wb+') as f:
        pickle.dump(episode_rewards, f)

#restart()


def restart2():
    plot_rewards = []
    plot_k = []
    num_of_decrease = 0
    previous_reward = None
    theta_k = None
    num_cut = 0
    m = 1
    i_iter = 0
    flag = True
    num_episodes = []
    num_episode = 0

    episode_rewards = []

    num_steps = 0
    total_cut = []
    temp = 0

    sta = []

    while num_steps < args.max_step_num:
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        if previous_reward == None:
            previous_reward = log['avg_reward']
            #save the first model parameters
            i_iter += 1
            #plot_k.append(num_episodes)
            plot_k.append(num_steps)
            num_episode += log['num_episodes']
            num_episodes.append(num_episode)
            plot_rewards.append(previous_reward)
            num_steps += log['num_steps']
            episode_rewards += log['episode_rewards']
            theta_k = policy_net.state_dict()
            flag = True
        else:
            #if J(theta_k) > J(theta _k+1):
            #       back to theta_k
            #       do not update parameters
            #       keep previous_reward unchanged
            #       increase num_restart by 1
            possible_thetas = [policy_net.state_dict()]
            average_rewards = [log['avg_reward']]
            cross_steps = [log['num_steps']]
            cross_episodes = [log['num_episodes']]
            cross_batch_episode_rewards = [log['episode_rewards']]

            batches = [batch]
            temp = [previous_reward]
            if previous_reward > log['avg_reward']:
                #load cut theta
                #print(cut_theta)
                policy_net.load_state_dict(cut_theta)
                num_cut += 1
                #generate trajectories with new theta to see if its better
                batch, log = agent.collect_samples(args.min_batch_size)
                batches.append(batch)
                possible_thetas.append(cut_theta)
                average_rewards.append(log['avg_reward'])
                cross_steps.append(log['num_steps'])
                cross_episodes.append(log['num_episodes'])
                cross_batch_episode_rewards.append(log['episode_rewards'])

                total_cut.append(num_cut)
                #     print("Restart, Reward: "+str(num_restart)+', '+str(previous_reward))
                num_cut = 0
                if previous_reward > log['avg_reward']:
                    flag = False
                    num_of_decrease += 1
                #     break
            
            #   accept theta_k+1, save its parameters
            #   update previous_reward
            if not flag:
                diff = (previous_reward - max(average_rewards))/abs(previous_reward)
                sta.append(diff)
                
                previous_reward = max(average_rewards)
                index = average_rewards.index(previous_reward)

                epi = cross_episodes[index]
                theta_k = possible_thetas[index]
                steps = cross_steps[index]

                current_batch_episode_rewards = cross_batch_episode_rewards[index]
                episode_rewards += current_batch_episode_rewards

                other_steps = np.sum(cross_steps) - steps
                batch = batches[index]

                i_iter += 1
                #plot_k.append(num_episodes)

                num_episode += epi
                num_episodes.append(num_episode)

                num_steps += other_steps
                plot_k.append(num_steps)
                plot_rewards.append(previous_reward)
                num_steps += steps
                flag = True
            else:
                theta_k = policy_net.state_dict()
                
                previous_reward = log['avg_reward']
                i_iter += 1
                #plot_k.append(num_episodes)
                current_batch_episode_rewards = log['episode_rewards']
                episode_rewards += current_batch_episode_rewards
                
                other_steps = np.sum(cross_steps) - log['num_steps']
                num_steps += other_steps
                plot_k.append(num_steps)

                num_episode += log['num_episodes']
                num_episodes.append(num_episode)

                plot_rewards.append(previous_reward)

                num_steps += log['num_steps']

        
        t0 = time.time()
        cut_theta = update_params_cut(batch, num_steps, 4)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], previous_reward))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    #with open(args.env_name+ '_vanilla'+'.pickled', 'rb') as f:
    #    object_file = list(pickle.load(f))
    #    plot_x_v = [x for x in object_file[0]]
    #    plot_r_v = [y for y in object_file[1]]

    #plt.plot(plot_k, plot_rewards)
    #plt.plot(plot_x_v, plot_r_v, label="vanilla")
    #plt.xlabel('Num_Steps')
    #plt.ylabel('Average reward over the trajectories')

    mean = np.mean(sta)
    std = np.std(sta)
    median = np.median(sta)

    title = args.env_name + '_restart2_'+ str(args.max_step_num) + "_" + str(args.seed)
    #notes = str(np.sum(total_cut)) + '/' + str(np.sum(total_cut) + i_iter - len(total_cut)) + ' mean:'+str(round(mean, 6)) + " std:" + str(round(std, 6)) + " median:" + str(round(median, 6))
    
    #plt.title(title + '\n' + notes)
    #plt.legend()
    #plt.savefig('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/images/'+ title + '.png')

    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ title +'.pickled', 'wb+') as f:
        pickle.dump([plot_k,plot_rewards], f)

    statistics = args.env_name + "_r2_" + str(args.max_step_num) + "_stat_" + str(args.seed)
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ statistics +'.pickled', 'wb+') as f:
        pickle.dump(sta, f)

    ep = args.env_name + "_" + str(args.max_step_num) + "_r2_episode_" + str(args.seed) 
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ ep +'.pickled', 'wb+') as f:
        pickle.dump(num_episodes, f)

    epr = args.env_name + "_" + str(args.max_step_num) + "_r2_episode_rewards_" + str(args.seed) 
    with open('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/pickle/'+ epr +'.pickled', 'wb+') as f:
        pickle.dump(episode_rewards, f)

#restart2()

def vary_samples():
    plot_rewards = []
    plot_k = []
    num_of_decrease = 0
    previous_reward = None
    num_episodes = []
    episode_rewards = []
    num_episode = 0

    episode_rewards = []

    divide_ratio = args.divide_ratio
    diff_ratio = args.diff_ratio
    diff_epochs = args.diff_epochs

    num_steps = 0
    i_iter = 0
    while num_steps < args.max_step_num/diff_ratio:
        """generate multiple trajectories that reach the minimum batch_size"""
        print(args.min_batch_size/divide_ratio)
        batch, log = agent.collect_samples(args.min_batch_size/divide_ratio)
        i_iter += 1
        #plot_k.append(num_episodes)

        if previous_reward == None:
            previous_reward = log['avg_reward']
        else:
            if previous_reward > log['avg_reward']:
                num_of_decrease += 1
            previous_reward = log['avg_reward']

        num_episode += log['num_episodes']
        num_episodes.append(num_episode)
        #print (log['episode_rewards'])
        episode_rewards += log['episode_rewards']

        plot_k.append(num_steps)
        plot_rewards.append(previous_reward)
        num_steps +=log['num_steps']

        t0 = time.time()
        update_params(batch, num_steps, diff_epochs)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()

    while args.max_step_num/diff_ratio <= num_steps and num_steps < args.max_step_num:

        batch, log = agent.collect_samples(args.min_batch_size)
        i_iter += 1
        #plot_k.append(num_episodes)
        #num_episodes += log['num_episodes']
        if previous_reward == None:
            previous_reward = log['avg_reward']
        else:
            if previous_reward > log['avg_reward']:
                num_of_decrease += 1
            previous_reward = log['avg_reward']

        num_episode += log['num_episodes']
        num_episodes.append(num_episode)
        episode_rewards += log['episode_rewards']

        plot_k.append(num_steps)
        plot_rewards.append(previous_reward)
        num_steps +=log['num_steps']

        t0 = time.time()
        update_params(batch, num_steps, 10)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()


        """clean up gpu memory"""
        
        torch.cuda.empty_cache()

    #plt.plot(plot_k, plot_rewards)
    #plt.xlabel('Num_Steps')
    #plt.ylabel('Average reward over the trajectories')

    title = args.env_name + '_varied_samples_' + str(args.max_step_num) + '_' + str(args.seed) + "_no_annealing_with_divide_ratio_" + str(divide_ratio) + "_diff_ratio_" + str(diff_ratio) + "_diff_epochs_" + str(diff_epochs)
    #plt.title(title)
    #plt.savefig('/Users/autumn/Academics/Research/PPO/PPO_Rejection/PyTorch-RL/images/'+ title + '.png')

    with open('/gpfsnyu/home/yw1370/PPO_Rejection/PyTorch-RL/pickle/'+ title +'.pickled', 'wb+') as f:
        pickle.dump([plot_k,plot_rewards], f)

    with open('/gpfsnyu/home/yw1370/PPO_Rejection/PyTorch-RL/pickle/'+ title +'_episode' +'.pickled', 'wb+') as f:
        pickle.dump(num_episodes, f)

    with open('/gpfsnyu/home/yw1370/PPO_Rejection/PyTorch-RL/pickle/'+ title +'_episode_rewards' +'.pickled', 'wb+') as f:
        pickle.dump(episode_rewards, f)


#vary_samples()

#restart percentage
#calculate [J(pi_k) - J(pi_k+1)] / J(pi_k) mean, std, medium for all the numbers
#at the end, different experiments, then calculate mean, max, min, average inprovement, same number of sample points