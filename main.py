import sys
import torch.cuda
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from model import *
from tools import *
from envs import make_vec_envs
import numpy as np
import random
from train_tools import train_tools
from tensorboardX import SummaryWriter
from tools import get_args, registration_envs
import gym
from memory import ReplayMemory
from rlagent import Agent

def main(args):

    # The name of this experiment, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\experiment' and '.\logs\runs'
    custom = input('Please input the experiment name\n')
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)
    
    print('Using device:', device)

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args, None)
    log_writer_path = './logs/runs/{}'.format('PCT-' + timeStr)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    tmp_env = gym.make(args.id, args=args)
    args.action_space = tmp_env.space.get_action_space()
    args.obs_len = tmp_env.observation_space.shape[0]

    # Create parallel packing environments to collect training samples online
    envs = make_vec_envs(args, './logs/runinfo', True)

    # Create the Rainbow DQN agent and replay memory
    DQN_agent = Agent(args)
    mem_num = args.num_processes
    mem_capacity = int(args.memory_capacity / mem_num)
    memory = [ReplayMemory(args=args, capacity=mem_capacity, obs_len=tmp_env.observation_space.shape[0]) for _ in range(mem_num)]



    # Load the trained model, if needed
    if args.load_model:
        PCT_policy = load_policy(args.model_path, PCT_policy)
        print('Loading pre-train model', args.model_path)

    # Perform all training.
    trainTool = train_tools(writer, timeStr, PCT_policy, args)
    trainTool.train_n_steps(envs, args, device)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)

