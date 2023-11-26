import argparse
import givenData
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Rainbow BPP arguments')

    # arguments for PCT
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    parser.add_argument('--lnes', type=str, default='EMS', help='Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC')
    parser.add_argument('--internal-node-holder', type=int, default=80, help='Maximum number of internal nodes')
    parser.add_argument('--leaf-node-holder', type=int, default=50, help='Maximum number of leaf nodes')
    parser.add_argument('--shuffle',type=bool, default=True, help='Randomly shuffle the leaf nodes')
    parser.add_argument('--continuous', action='store_true', help='Use continuous enviroment, otherwise the enviroment is discrete')

    parser.add_argument('--no-cuda',action='store_true', help='Forbidden cuda')
    parser.add_argument('--device', type=int, default=0, help='Which GPU will be called')
    parser.add_argument('--seed',   type=int, default=4, help='Random seed')

    # arguments for RL training
    parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=31, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-1, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=8, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1.0, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--target-update', type=int, default=int(1e3), metavar='τ', 
                        help='Number of steps after which to update target network')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--learn-start', type=int, default=int(5e2), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--num-processes', type=int, default=8, help='The number of parallel processes used for training')
    parser.add_argument('--multi-step', type=int, default=3, help='The rollout length for n-step training')
    parser.add_argument('--learning-rate', type=float, default=1e-6, metavar='η', help='Learning rate, only works for A2C')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--max-grad-norm',          type=float, default=0.5, help='Max norm of gradients')
    parser.add_argument('--embedding-size',     type=int, default=64,  help='Dimension of input embedding')
    parser.add_argument('--hidden-size',        type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--gat-layer-num',      type=int, default=1, help='The number GAT layers')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='γ', help='Discount factor')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')

    parser.add_argument('--save-interval', default=1000, help='How often to save the model.')
    parser.add_argument('--checkpoint-interval', default=20000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--model-update-interval',  type=int,   default=20e30 , help='How often to create a new model')
    parser.add_argument('--model-save-path',type=str, default='./logs/experiment', help='The path to save the trained model')
    parser.add_argument('--print-log-interval',     type=int,   default=30, help='How often to print training logs')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes', type=int, default=100, metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--load-model', action='store_true', help='Load the trained model')
    parser.add_argument('--model-path', type=str, help='The path to load model')
    parser.add_argument('--load-dataset', action='store_true', help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')

    parser.add_argument('--sample-from-distribution', action='store_true', help='Sample continuous item size from a uniform distribution U(a,b), otherwise sample items from \'item_size_set\' in \'givenData.py\'')
    parser.add_argument('--sample-left-bound', type=float, metavar='a', help='The parametre a of distribution U(a,b)')
    parser.add_argument('--sample-right-bound', type=float, metavar='b', help='The parametre b of distribution U(a,b)')

    args = parser.parse_args()

    if args.no_cuda: args.device = 'cpu'

    args.container_size = givenData.container_size
    args.item_size_set  = givenData.item_size_set

    if args.sample_from_distribution and args.sample_left_bound is None:
        args.sample_left_bound = 0.1 * min(args.container_size)
    if args.sample_from_distribution and args.sample_right_bound is None:
        args.sample_right_bound = 0.5 * min(args.container_size)

    if args.continuous:
        args.id = 'PctContinuous-v0'
    else:
        args.id = 'PctDiscrete-v0'

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    if args.evaluate:
        args.num_processes = 1
    args.norm_factor = 1.0 / np.max(args.container_size)

    args.model_path = './evaltest.pt'
    args.save_memory_path = None

    return args

def get_args_heuristic():
    parser = argparse.ArgumentParser(description='Heuristic baseline arguments')

    parser.add_argument('--continuous', action='store_true', help='Use continuous enviroment, otherwise the enviroment is discrete')
    parser.add_argument('--setting', type=int, default=2, help='Experiment setting, please see our paper for details')
    # parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of episodes evaluated')
    parser.add_argument('--load-dataset', action='store_true', help='Load an existing dataset, otherwise the data is generated on the fly')
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')
    parser.add_argument('--heuristic', type=str, default='LSAH', help='Options: LSAH DBL MACS OnlineBPH HM BR RANDOM')


    args = parser.parse_args()
    args.container_size = givenData.container_size
    args.item_size_set  = givenData.item_size_set
    args.evaluate = True

    if args.continuous:
        assert args.heuristic == 'LSAH' or args.heuristic == 'OnlineBPH' or args.heuristic == 'BR', 'only LSAH, OnlineBPH, and BR allowed for continuous environment'

    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    if args.evaluate:
        args.num_processes = 1
    args.normFactor = 1.0 / np.max(args.container_size)

    return args