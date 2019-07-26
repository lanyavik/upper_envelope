import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time
import random

def collect_samples(pid, queue, env, policy, custom_reward, mean_action,
                    tensor, render, running_state, update_rs, min_batch_size,seed,thread_id,early_stopping=False):
    torch.randn(pid, )
    log = dict()
    if early_stopping:
        training = Memory()
        validation = Memory()
        memory = Memory()
    else:
        memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    episode_rewards = []

    while num_steps < min_batch_size:
        #env.seed(seed + thread_id)
        state = env.reset()

        #print("state after env.reset():",state)

        if running_state is not None:
            state = running_state(state, update=update_rs)
        reward_episode = 0

        #
        for t in range(min_batch_size - num_steps):
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            if mean_action:
                action = policy(state_var)[0].data[0].numpy()
            else:
                action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            #print("action:",action)
            #get env, action and then get reward and next_state
            next_state, reward, done, _ = env.step(action)
            #reward sum of this episode
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state, update=update_rs)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            if early_stopping:
                ra = random.random()
                if ra > 0.8 and len(validation) <= min_batch_size*0.1:
                    validation.push(state, action, mask, next_state, reward)
                else:
                    training.push(state, action, mask, next_state, reward)
            else:
                memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        # log stats
        #print ('episode length:',t)
        num_steps += (t + 1)
        # num_episode = num_trajectories
        num_episodes += 1
        total_reward += reward_episode
        episode_rewards.append(reward_episode)
        #print("total_reward:", total_reward)
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
    #print (num_steps)

    log['num_steps'] = min_batch_size
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['episode_rewards'] = episode_rewards
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        if early_stopping:
            return memory, training, validation, log
        else:
            return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['episode_rewards'] = list(np.concatenate([x['episode_rewards'] for x in log_list]))
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    print ('num_episodes:',log['num_episodes'])
    print ('len of episode_rewards:', len(log['episode_rewards']))
    return log


class Agent:

    def __init__(self, env_factory, args, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, running_state=None, num_threads=1,seed=0,thread_id=0, early_stopping = False):
        self.env_factory = env_factory
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.env_list = []
        self.seed = seed
        self.thread_id = thread_id
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i, args))

        self.early_stopping = early_stopping

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            self.policy.cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, self.running_state, False, thread_batch_size, self.seed, self.thread_id)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        if self.early_stopping:
            memory, training, validation, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size, self.seed, self.thread_id, self.early_stopping)

        else:
            memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size, self.seed, self.thread_id, self.early_stopping)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)


        if self.early_stopping:
            memory.append(training)
            memory.append(validation)
            training = training.sample()
            validation = validation.sample()
            batch = memory.sample()
        else:
            batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            self.policy.cuda()
        t_end = time.time()
        
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        #state action mask reward next_state
        if self.early_stopping:
            return training, validation, log
        else:
            return batch, log
