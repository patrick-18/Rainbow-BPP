import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from tools import *
from evaluation_tools import evaluate
import gym
import random
from rlagent import Agent
from arguments import get_args

def main(args):
    # The name of this evaluation, related file backups and experiment tensorboard logs will
    # be saved to '.\logs\evaluation' and '.\logs\runs'
    custom = input('Please input the evaluate name\n')
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

    # Create single packing environment and load existing dataset.
    envs = gym.make(args.id, args=args)
    args.action_space = args.leaf_node_holder
    args.obs_len = envs.observation_space.shape[0]

    # Create the Rainbow DQN agent
    DQN_agent = Agent(args)

    # Backup all py file
    backup(timeStr, args, None)
    
    # Perform all evaluation.
    evaluate(DQN_agent, envs, timeStr, args, device,
             eval_episode=args.evaluation_episodes)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    args.evaluate = True
    args.num_processes = 1
    args.load_model = True
    main(args)
