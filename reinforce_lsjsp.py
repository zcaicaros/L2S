import numpy as np
from parameters import args

import torch
import torch.optim as optim
from env.env_single import JsspN5
from model.actor import Actor
from env.generateJSP import uni_instance_gen
from torch_geometric.data.batch import Batch

torch.manual_seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

env = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, transition=args.transit, init='rule', rule='spt')
policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)  # policy = Actor(3, 64, gin_l=3, policy_l=3).to(dev)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()


def finish_episode(rewards, log_probs):
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if returns.shape[0] > 1:  # normalization should be over >=2 elements otherwise it will get Nan and causing error.
        returns = (returns - returns.mean()) / (torch.std(returns) + eps)

    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


def main():
    running_reward = 0
    incumbent_reward = -100000
    log = []
    np.random.seed(2)
    # instance = uni_instance_gen(args.j, args.m, args.l, args.h)
    # np.save('./instance.npy', instance)
    for i_episode in range(1, args.episodes + 1):
        instance = uni_instance_gen(args.j, args.m, args.l, args.h)
        state, feasible_action, done = env.reset(instance=instance, fix_instance=True)
        ep_reward = 0
        rewards = []
        log_probs = []
        if not done:  # if init state is done then do nothing
            while not done:
                action, log_p = policy(Batch.from_data_list([state]).to(dev), [feasible_action])
                state, reward, feasible_action, done = env.step_single(action[0])
                rewards.append(reward)
                log_probs.append(log_p)
                ep_reward += reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            # finish_episode(rewards, log_probs)
        log.append([env.current_objs, ep_reward, running_reward])
        if i_episode % 100 == 0:
            np.save('log/log_10x10_sample_25.6w_spt_no-learn.npy', np.array(log))
        print('solution quality:', env.current_objs)
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))
        if running_reward > incumbent_reward:
            print('better running reward, saving network...')
            torch.save(policy.state_dict(), './{}.pth'.format(str(args.j) + '_' + str(args.m)))
            incumbent_reward = running_reward
        print()


if __name__ == '__main__':
    main()