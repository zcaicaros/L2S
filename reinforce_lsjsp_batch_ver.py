import time

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


def finish_episode(rewards, log_probs, dones):

    R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
    returns = []
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.cat(returns, dim=-1)
    dones = torch.cat(dones, dim=-1)
    log_probs = torch.cat(log_probs, dim=-1)


    losses = []
    for b in range(returns.shape[0]):
        masked_R = torch.masked_select(returns[b], ~dones[b])
        masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R) + eps)
        masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
        loss = (- masked_log_prob * masked_R).sum()
        losses.append(loss)

    optimizer.zero_grad()
    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()
    optimizer.step()


def main():

    batch_size = 48
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
    # validation_data = np.load('./validation_data/validation_instance_{}x{}.npy'.format(args.j, args.m))
    np.random.seed(1)

    # instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(batch_size)])  # fixed instances
    # np.save('./instances.npy', instances)
    for batch_i in range(1, args.episodes // batch_size + 1):

        t1 = time.time()

        instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(batch_size)])
        states, feasible_actions, dones = env.reset(instances=instances, init_type=init, device=dev)
        batch_data.wrapper(*states)

        ep_reward_log = []
        rewards_buffer = []
        log_probs_buffer = []
        dones_buffer = [dones]

        while env.itr < args.transit:
            # print(len(feasible_actions))
            actions, log_ps = policy(batch_data, feasible_actions)
            states, rewards, feasible_actions, dones = env.step(actions, dev)
            batch_data.wrapper(*states)

            # store training data
            rewards_buffer.append(rewards)
            log_probs_buffer.append(log_ps)
            dones_buffer.append(dones)

            # logging...
            ep_reward_log.append(rewards)

            '''if env.itr % 10 == 0:
                # training...
                finish_episode(rewards_buffer, log_probs_buffer, dones_buffer[:-1])
                ep_reward_log = []
                rewards_buffer = []
                log_probs_buffer = []
                dones_buffer = [dones]'''

        finish_episode(rewards_buffer, log_probs_buffer, dones_buffer[:-1])

        t2 = time.time()
        print('Batch {} training takes: {:.2f}'.format(batch_i, t2 - t1),
              'Mean Performance: {:.2f}'.format(env.current_objs.cpu().mean().item()))
        log.append(env.current_objs.cpu().mean().item())
        np.save('./log/batch_log_{}x{}_{}w_{}_{}.npy'.format(args.j, args.m, args.episodes / 10000, init, str(args.transit)),
                np.array(log))



if __name__ == '__main__':

    main()

    print()