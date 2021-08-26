import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops


class DGHANlayer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, dropout, concat, heads=2):
        super(DGHANlayer, self).__init__()
        self.dropout = dropout
        self.opsgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)
        self.mchgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)

    def forward(self, node_h, edge_index_pc, edge_index_mc):
        node_h_pc = F.elu(self.opsgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_pc))
        node_h_mc = F.elu(self.mchgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_mc))
        node_h = torch.mean(torch.stack([node_h_pc, node_h_mc]), dim=0, keepdim=False)
        return node_h


class DGHAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layer_dghan=4, heads=2):
        super(DGHAN, self).__init__()
        self.layer_dghan = layer_dghan
        self.hidden_dim = hidden_dim

        ## DGHAN conv layers
        self.DGHAN_layers = torch.nn.ModuleList()

        # init DGHAN layer
        if layer_dghan == 1:
            # only DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=False, heads=heads))
        else:
            # first DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=True, heads=heads))
            # following DGHAN layers
            for layer in range(layer_dghan - 2):
                self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=True, heads=heads))
            # last DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=False, heads=1))

    def forward(self, x, edge_index_pc, edge_index_mc, batch_size):

        # initial layer forward
        h_node = self.DGHAN_layers[0](x, edge_index_pc, edge_index_mc)
        for layer in range(1, self.layer_dghan):
            h_node = self.DGHAN_layers[layer](h_node, edge_index_pc, edge_index_mc)

        return h_node, torch.mean(h_node.reshape(batch_size, -1, self.hidden_dim), dim=1)


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

    def forward(self, x, edge_index, batch):

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
                 embedding_l=4,
                 policy_l=3,
                 embedding_type='gin',
                 heads=4,
                 dropout=0.6):
        super(Actor, self).__init__()
        self.embedding_l = embedding_l
        self.policy_l = policy_l
        self.embedding_type = embedding_type
        if self.embedding_type == 'gin':
            self.embedding = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
        elif self.embedding_type == 'dghan':
            self.embedding = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, heads=heads)
        elif self.embedding_type == 'gin+dghan':
            self.embedding_gin = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
            self.embedding_dghan = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, heads=heads)
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        # policy
        self.policy = torch.nn.ModuleList()
        if policy_l == 1:
            if self.embedding_type == 'gin+dghan':
                self.policy.append(Sequential(Linear(hidden_dim * 4, hidden_dim),
                                              # torch.nn.BatchNorm1d(hidden_dim),
                                              torch.nn.Tanh(),
                                              Linear(hidden_dim, hidden_dim)))
            else:
                self.policy.append(Sequential(Linear(hidden_dim * 2, hidden_dim),
                                              # torch.nn.BatchNorm1d(hidden_dim),
                                              torch.nn.Tanh(),
                                              Linear(hidden_dim, hidden_dim)))
        else:
            for layer in range(policy_l):
                if layer == 0:
                    if self.embedding_type == 'gin+dghan':
                        self.policy.append(Sequential(Linear(hidden_dim * 4, hidden_dim),
                                                      # torch.nn.BatchNorm1d(hidden_dim),
                                                      torch.nn.Tanh(),
                                                      Linear(hidden_dim, hidden_dim)))
                    else:
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

        if self.embedding_type == 'gin':
            node_embed, graph_embed = self.embedding(batch_states.x,
                                                     add_self_loops(torch.cat([batch_states.edge_index_pc,
                                                                               batch_states.edge_index_mc],
                                                                              dim=-1))[0],
                                                     batch_states.batch)
        elif self.embedding_type == 'dghan':
            node_embed, graph_embed = self.embedding(batch_states.x,
                                                     add_self_loops(batch_states.edge_index_pc)[0],
                                                     add_self_loops(batch_states.edge_index_mc)[0],
                                                     len(feasible_actions))
        elif self.embedding_type == 'gin+dghan':
            node_embed_gin, graph_embed_gin = self.embedding_gin(batch_states.x,
                                                                 add_self_loops(torch.cat([batch_states.edge_index_pc,
                                                                                           batch_states.edge_index_mc],
                                                                                          dim=-1))[0],
                                                                 batch_states.batch)
            node_embed_dghan, graph_embed_dghan = self.embedding_dghan(batch_states.x,
                                                                       add_self_loops(batch_states.edge_index_pc)[0],
                                                                       add_self_loops(batch_states.edge_index_mc)[0],
                                                                       len(feasible_actions))
            node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)
            graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1)
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        device = node_embed.device
        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0] // batch_size

        # augment node embedding with graph embedding then forwarding policy
        node_embed_augmented = torch.cat([node_embed, graph_embed.repeat_interleave(repeats=n_nodes_per_state, dim=0)], dim=-1).reshape(batch_size, n_nodes_per_state, -1)
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

        dist = Categorical(probs=pi)
        actions_id = dist.sample()
        # actions_id = torch.argmax(pi, dim=-1)  # greedy action
        sampled_actions = [[actions_id[i].item() // n_nodes_per_state, actions_id[i].item() % n_nodes_per_state] for i in range(len(feasible_actions))]
        log_prob = dist.log_prob(actions_id)
        return sampled_actions, log_prob


if __name__ == '__main__':
    import random
    from env.env_batch import JsspN5, BatchGraph
    from env.generateJSP import uni_instance_gen

    dev = 'cpu' if torch.cuda.is_available() else 'cpu'

    n_j = 150
    n_m = 25
    l = 1
    h = 99
    reward_type = 'yaoxin'
    init_type = 'fdd-divide-mwkr'
    b_size = 1
    transit = 2000
    n_workers = 5
    hid_dim = 4

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    env = JsspN5(n_job=n_j, n_mch=n_m, low=l, high=h, reward_type=reward_type)
    batch_data = BatchGraph()
    instances = np.load('../test_data/syn{}x{}.npy'.format(n_j, n_m))
    # instances = np.array([uni_instance_gen(n_j=n_j, n_m=n_m, low=l, high=h) for _ in range(b_size)])
    states, feasible_as, dones = env.reset(instances=instances, init_type=init_type, device=dev)
    print(env.incumbent_objs)
    # print(feasible_as)

    actor = Actor(3, hid_dim, embedding_l=4, policy_l=4, embedding_type='gin+dghan').to(dev)
    while env.itr < transit:
        batch_data.wrapper(*states)
        actions, log_ps = actor(batch_data, feasible_as)
        states, rewards, feasible_as, dones = env.step(actions, dev)
        # print(actions)
        # print(env.incumbent_objs)
        # print(feasible_as)

    # grad = torch.autograd.grad(log_ps.sum(), [param for param in actor.parameters()])