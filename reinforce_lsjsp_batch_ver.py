import numpy as np
from parameters import args

import torch
import torch.optim as optim
from env.env_batch import JsspN5
from model.actor import Actor
from env.generateJSP import uni_instance_gen
from torch_geometric.data.batch import Batch

torch.manual_seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

init = 'fdd-divide-mwkr'
env = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, transition=args.transit)
env_validation = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, transition=args.transit)
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

    batch_size = 4
    from env.env_batch import BatchGraph
    batch_data = BatchGraph()

    running_reward = 0
    incumbent_validation_result = np.inf
    log = []
    validation_log = []
    # remember to generate validation data if size is not in {10x10, 15x15, 20x20, 30x20}
    # np.random.seed(2)
    # validation_data = np.array([uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)])
    # np.save('./validation_data/validation_instance_{}x{}.npy'.format(args.j, args.m), validation_data)
    validation_data = np.load('./validation_data/validation_instance_{}x{}.npy'.format(args.j, args.m))
    np.random.seed(1)

    # instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(batch_size)])  # fixed instances
    # np.save('./instances.npy', instances)
    for i_episode in range(1, args.episodes // batch_size + 1):
        instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(batch_size)])
        states, feasible_actions, dones = env.reset(instances=instances, init_type=init, device=dev)
        batch_data.wrapper(*states)

        ep_reward_log = []
        rewards_buffer = []
        log_probs_buffer = []
        while env.itr < args.transit:
            actions, log_ps = policy(batch_data, feasible_actions)
            states, rewards, feasible_actions, dones = env.step(actions, dev)
            batch_data.wrapper(*states)

            # store training data
            rewards_buffer.append(rewards)
            log_probs_buffer.append(log_ps)

            # logging...
            ep_reward_log.append(rewards)

        # training...
        finish_episode(rewards_buffer, log_probs_buffer)


if __name__ == '__main__':
    main()