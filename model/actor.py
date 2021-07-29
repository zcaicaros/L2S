import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_gin=4):
        super(GIN, self).__init__()
        self.layer_gin = layer_gin

        ## GIN conv layers
        self.GIN_layers = torch.nn.ModuleList()

        # init gin layer
        self.GIN_layers.append(
            GINConv(
                Sequential(Linear(in_dim, hidden_dim),
                           torch.nn.BatchNorm1d(hidden_dim),
                           ReLU(),
                           Linear(hidden_dim, hidden_dim)),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target")
        )

        # rest gin layers
        for layer in range(layer_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim),
                               torch.nn.BatchNorm1d(hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim)),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target")
            )

    def forward(self, batch_states):

        x, edge_index, batch = batch_states.x, batch_states.edge_index, batch_states.batch
        # x, edge_index, batch = batch_states[0], batch_states[1], batch_states[2]

        # print(x)
        # print(torch_geometric.utils.sort_edge_index(edge_index)[0])
        # print(batch)

        hidden_rep = []
        node_pool_over_layer = 0
        # initial layer forward
        h = self.GIN_layers[0](x, edge_index)
        node_pool_over_layer += h
        hidden_rep.append(h)
        # rest layers forward
        for layer in range(1, self.layer_gin):
            h = self.GIN_layers[layer](h, edge_index)
            node_pool_over_layer += h
            hidden_rep.append(h)

        # Graph pool
        gPool_over_layer = 0
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += g_pool

        return node_pool_over_layer, gPool_over_layer


class Actor(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 gin_l=4,
                 policy_l=3):
        super(Actor, self).__init__()
        # self.gin_l = gin_l
        self.policy_l = policy_l

        self.embedding = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=gin_l)

        # policy
        self.policy = torch.nn.ModuleList()
        if policy_l == 1:
            self.policy.append(Sequential(Linear(hidden_dim * 2, hidden_dim),
                                          # torch.nn.BatchNorm1d(hidden_dim),
                                          torch.nn.Tanh(),
                                          Linear(hidden_dim, hidden_dim)))
        else:
            for layer in range(policy_l):
                if layer == 0:
                    self.policy.append(Sequential(Linear(hidden_dim * 2, hidden_dim),
                                                  # torch.nn.BatchNorm1d(hidden_dim),
                                                  torch.nn.Tanh(),
                                                  Linear(hidden_dim, hidden_dim)))
                else:
                    self.policy.append(Sequential(Linear(hidden_dim, hidden_dim),
                                                  # torch.nn.BatchNorm1d(hidden_dim),
                                                  torch.nn.Tanh(),
                                                  Linear(hidden_dim, hidden_dim)))

    def forward(self, batch_states, feasible_actions):

        node_embed, graph_embed = self.embedding(batch_states)

        # print(graph_embed)
        # torch.save(graph_embed[0].unsqueeze(0), 'C:/Users/CONG030/Desktop/reinforce_debug/compare/actor.pt')

        device = node_embed.device
        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0] // batch_size

        # augment node embedding with graph embedding then forwarding policy
        node_embed_augmented = torch.cat([node_embed, graph_embed.repeat_interleave(repeats=n_nodes_per_state, dim=0)],
                                         dim=-1).reshape(batch_size, n_nodes_per_state, -1)
        for layer in range(self.policy_l):
            node_embed_augmented = self.policy[layer](node_embed_augmented)

        # action score
        action_score = torch.bmm(node_embed_augmented, node_embed_augmented.transpose(-1, -2))

        # prepare mask
        carries = np.arange(0, batch_size * n_nodes_per_state, n_nodes_per_state)
        a_merge = []  # merge index of actions of all states
        action_count = []  # list of #actions for each state
        for i in range(len(feasible_actions)):
            action_count.append(len(feasible_actions[i]))
            for j in range(len(feasible_actions[i])):
                a_merge.append([feasible_actions[i][j][0] + carries[i], feasible_actions[i][j][1]])
        a_merge = np.array(a_merge)
        mask = torch.ones(size=[batch_size * n_nodes_per_state, n_nodes_per_state], dtype=torch.bool, device=device)
        mask[a_merge[:, 0], a_merge[:, 1]] = False
        mask.resize_as_(action_score)

        # pi
        action_score.masked_fill_(mask, -np.inf)
        action_score_flat = action_score.reshape(batch_size, 1, -1)
        pi = F.softmax(action_score_flat, dim=-1)

        # print(torch.where(pi[0] != 0))
        # print(pi[0][torch.where(pi[0] != 0)])
        # print(feasible_actions[0])

        dist = Categorical(probs=pi)
        actions_id = dist.sample()
        # actions_id = torch.argmax(pi, dim=-1)  # greedy action
        sampled_actions = [[actions_id[i].item() // n_nodes_per_state, actions_id[i].item() % n_nodes_per_state] for i
                           in range(len(feasible_actions))]
        log_prob = dist.log_prob(actions_id)
        return sampled_actions, log_prob


if __name__ == '__main__':
    import random
    from env.env_batch import JsspN5, BatchGraph
    from env.generateJSP import uni_instance_gen

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_j = 10
    n_m = 10
    l = 1
    h = 99
    reward_type = 'yaoxin'
    init_type = 'fdd-divide-mwkr'
    b_size = 2
    transit = 1
    n_workers = 5
    hid_dim = 4

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    env = JsspN5(n_job=n_j, n_mch=n_m, low=l, high=h, reward_type=reward_type)
    batch_data = BatchGraph()
    embedding = GIN(in_dim=3, hidden_dim=hid_dim, layer_gin=3).to(dev)
    actor = Actor(3, hid_dim, gin_l=3, policy_l=3).to(dev)

    instances = np.array([uni_instance_gen(n_j=n_j, n_m=n_m, low=l, high=h) for _ in range(b_size)])
    states, feasible_as, dones = env.reset(instances=instances, init_type=init_type, device=dev)

    while env.itr < transit:
        batch_data.wrapper(*states)
        actions, log_ps = actor(batch_data, feasible_as)
        states, rewards, feasible_as, dones = env.step(actions, dev)

        print(actions)
