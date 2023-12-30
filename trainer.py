import os
import numpy as np
import torch
from tools import observation_decode_leaf_node, get_leaf_nodes, data_augmentation
from tqdm import trange
from collections import deque
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
np.set_printoptions(threshold=np.inf)
import time

# Distributed training for online packing policy
def learningPara(T, priority_weight_increase, model_save_path, dqn, mem, timeStr, args, counter, lock, sub_time_str):
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr + '-loss')
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    targetCounter = T
    checkCounter = T
    logCounter = T
    timeStep = T
    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print('Distributed Training Start')
    torch.set_num_threads(1)
    while True:
        if not lock.value:
            for i in range(len(mem)):
                mem[i].priority_weight = min(mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            dqn.reset_noise()
            loss = dqn.learn(mem)  # Train with n-step distributional double-Q learning

            # Update target network
            if timeStep - targetCounter >= args.target_update:
                targetCounter = timeStep
                dqn.update_target_net()

            if timeStep % args.checkpoint_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            # Checkpoint the network #
            if (args.checkpoint_interval != 0) and (timeStep - checkCounter >= args.save_interval):
                checkCounter = timeStep
                dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

            if timeStep - logCounter >= args.print_log_interval:
                logCounter = timeStep
                writer.add_scalar("Training/Value loss", loss.mean().item(), logCounter)

            timeStep += 1
        else:
            time.sleep(0.5)


class Trainer(object):
    def __init__(self, writer, timeStr, dqn, mem):
        self.writer = writer
        self.timeStr = timeStr
        self.dqn = dqn
        self.mem = mem

    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)


        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        state = envs.reset()

        batchX = torch.arange(args.num_processes)

        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, loss = 0, 0
        if args.distributed:
            counter= mp.Value('i', 0)
            lock = mp.Value('b', False)
        # Training loop
        self.dqn.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0 and not args.distributed:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            all_nodes, leaf_nodes = get_leaf_nodes(state,
                                                   internal_node_holder=args.internal_node_holder,
                                                   leaf_node_holder=args.leaf_node_holder)

            _, _, _, _, mask = observation_decode_leaf_node(all_nodes,
                                                            internal_node_holder=args.internal_node_holder,
                                                            leaf_node_holder=args.leaf_node_holder,
                                                            internal_node_length=args.internal_node_length)
            action = self.dqn.act(all_nodes, mask)  # Choose an action greedily (with noisy weights)
            selected_leaf_nodes = leaf_nodes[batchX, action.squeeze()]



            next_state, reward, done, infos = envs.step(selected_leaf_nodes.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])


            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            if not args.DA:
                for i in range(len(state)):
                    if validSample[i]:
                        self.mem[i].append(state[i], action[i], reward[i], done[i])  # Append transition to memory
            else:
                horizontal_flipped_state, vertical_flipped_state, center_flipped_state = data_augmentation(state, args.container_size)
                for i in range(len(state)):
                    if validSample[i]:
                        self.mem[i].append(state[i], action[i], reward[i], done[i])
                        self.mem[i].append(horizontal_flipped_state[i], action[i], reward[i], done[i])
                        self.mem[i].append(vertical_flipped_state[i], action[i], reward[i], done[i])
                        self.mem[i].append(center_flipped_state[i], action[i], reward[i], done[i])

            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    learningProcess = mp.Process(target=learningPara, args=(T, priority_weight_increase, model_save_path, self.dqn, self.mem, self.timeStr, args, counter, lock, sub_time_str))
                    learningProcess.start()
            else:
                # Train and test
                if T >= args.learn_start:
                    for i in range(len(self.mem)):
                        self.mem[i].priority_weight = min(self.mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    if T % args.replay_frequency == 0:
                        loss = self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
                    # Update target network
                    if T % args.target_update == 0:
                        self.dqn.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/Value loss",  loss.mean().item(), T)
                        if len(episode_rewards) != 0:
                            self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                            self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                            self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
                        if len(episode_ratio) != 0:
                            self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
                        if len(episode_counter) != 0:
                            self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

            if np.all(done): # Terminal state
                state = envs.reset()
            else:
                state = next_state
