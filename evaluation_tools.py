import os
import numpy as np
import torch
import tools


def evaluate(agent, eval_envs, timeStr, args, device, eval_episode = 100):
    agent.online_net.eval()
    agent.target_net.eval()
    obs = eval_envs.reset()
    obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
    all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                             args.internal_node_holder, args.leaf_node_holder)
    _, _, _, _, mask = tools.observation_decode_leaf_node(all_nodes,
                                                          internal_node_holder=args.internal_node_holder,
                                                          leaf_node_holder=args.leaf_node_holder,
                                                          internal_node_length=args.internal_node_length)
    batchX = torch.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []

    while step_counter < eval_episode:
        with torch.no_grad():
            action = agent.act(all_nodes, mask)
        selected_leaf_node = leaf_nodes[batchX, action.squeeze()]
        items = eval_envs.packed
        obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6])

        if done:
            print('Episode {} ends.'.format(step_counter))
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()

        obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                 args.internal_node_holder, args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
        _, _, _, _, mask = tools.observation_decode_leaf_node(all_nodes,
                                                              internal_node_holder=args.internal_node_holder,
                                                              leaf_node_holder=args.leaf_node_holder,
                                                              internal_node_length=args.internal_node_length)

    result = "Evaluation using {} episodes\n" \
             "Mean ratio {:.5f}, Var ratio {:.5f}, mean length {:.5f}\n".format(len(episode_ratio), np.mean(episode_ratio), np.var(episode_ratio) ,np.mean(episode_length))
    print(result)
    # Save the test trajectories.
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    # Write the test results into local file.
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()
