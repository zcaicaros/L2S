import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
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
        ## init projection
        # 1st mlp layer
        '''self.mlp1 = Sequential(Linear(in_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(),
                               Linear(hidden_dim, hidden_dim))'''

        ## GIN conv layers
        self.GIN_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # init gin layer
        nn_layer = Sequential(Linear(in_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.GIN_layers.append(GINConv(nn_layer, eps=0, train_eps=False, aggr='mean', flow="source_to_target"))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        for layer in range(layer_gin - 1):
            nn_layer = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.GIN_layers.append(GINConv(nn_layer, eps=0, train_eps=False, aggr='mean', flow="source_to_target"))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        ## layers used in graph pooling
        '''self.linear_prediction = torch.nn.ModuleList()
        for layer in range(self.layer_gin):
            self.linear_prediction.append(nn.Linear(hidden_dim, hidden_dim))'''

    def forward(self, batch_states):

        x, edge_index, batch = batch_states.x, batch_states.edge_index, batch_states.batch

        # init projection
        # h = self.mlp1(x)
        # hidden_rep = [h]

        ## GIN conv
        hidden_rep = []
        node_pool_over_layer = 0
        # initial layer forward
        h = self.batch_norms[0](F.relu(self.GIN_layers[0](x, edge_index)))
        node_pool_over_layer += h
        hidden_rep.append(h)
        for layer in range(1, self.layer_gin):
            h = self.batch_norms[layer](F.relu(self.GIN_layers[layer](h, edge_index)))
            node_pool_over_layer += h
            hidden_rep.append(h)

        gPool_over_layer = 0
        # Graph pool
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += g_pool
            '''if layer == 0:
                gPool_over_layer += g_pool
            else:
                gPool_over_layer += F.dropout(self.linear_prediction[layer-1](g_pool),
                                              0.5,
                                              training=self.training)'''

        return node_pool_over_layer, gPool_over_layer


class Actor(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 gin_l=4,
                 policy_l=3):
        super(Actor, self).__init__()
        self.gin_l = gin_l
        self.policy_l = policy_l

        self.embedding = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=gin_l)

        # policy
        self.policy = torch.nn.ModuleList()
        if policy_l == 1:
            self.policy.append(Sequential(Linear(hidden_dim * 3, hidden_dim),
                                          # torch.nn.BatchNorm1d(hidden_dim),
                                          torch.nn.ReLU(),
                                          Linear(hidden_dim, 1)))
        else:
            for layer in range(policy_l):
                if layer == 0:
                    self.policy.append(Sequential(Linear(hidden_dim * 3, hidden_dim),
                                                  # torch.nn.BatchNorm1d(hidden_dim),
                                                  torch.nn.ReLU(),
                                                  Linear(hidden_dim, int(hidden_dim / 2))))
                elif 0 < layer < (policy_l - 1):
                    self.policy.append(Sequential(Linear(int(hidden_dim / (2*layer)), int(hidden_dim / (2*layer))),
                                                  # torch.nn.BatchNorm1d(int(hidden_dim / (2*layer))),
                                                  torch.nn.ReLU(),
                                                  Linear(int(hidden_dim / (2*layer)), int(hidden_dim / (2*(layer+1))))))
                else:
                    self.policy.append(Sequential(Linear(int(hidden_dim / (2 * layer)), int(hidden_dim / (2 * layer))),
                                                  # torch.nn.BatchNorm1d(int(hidden_dim / (2 * layer))),
                                                  torch.nn.ReLU(),
                                                  Linear(int(hidden_dim / (2 * layer)), 1)))

    @staticmethod
    def sample_actions(scores, feasible_actions):
        scores = scores.squeeze()
        device = scores.device
        max_len = max([len(e) for e in feasible_actions])
        batch_size = len(feasible_actions)

        indices = [j + i * max_len for i in range(len(feasible_actions)) for j in range(len(feasible_actions[i]))]
        indices = torch.from_numpy(np.array(indices, dtype=np.int64)).to(device)
        scores_padded = torch.ones(batch_size * max_len, device=device) * (-np.inf)
        scores_padded[indices] = scores
        scores_padded = scores_padded.reshape(batch_size, -1)
        pi = F.softmax(scores_padded, dim=-1)
        dist = Categorical(probs=pi)
        actions_id = dist.sample()
        sampled_actions = [feasible_actions[i][actions_id[i].item()] for i in range(len(feasible_actions))]
        log_prob = dist.log_prob(actions_id)
        # print('a ID:', actions_id)
        # print('feasible a:', actions)
        # print('sampled a:', sampled_actions)
        # print('prob:', pi)
        # print('log prob:', log_prob)
        mean_entropy = dist.entropy().mean()  # mean entropy for batch
        # print(dist.probs)
        # print((- dist.probs * dist.probs.log()).sum(-1).mean().item())
        # print(entropy.item())
        return sampled_actions, log_prob, mean_entropy

    def forward(self, batch_states, feasible_actions):

        node_embed, graph_embed = self.embedding(batch_states)

        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0]//batch_size
        carries = np.arange(0, batch_size*n_nodes_per_state, n_nodes_per_state)

        # Prepare actions
        a_merge = []  # merge index of actions of all states
        action_count = []  # list of #actions for each state
        for i in range(len(feasible_actions)):
            action_count.append(len(feasible_actions[i]))
            for j in range(len(feasible_actions[i])):
                a_merge.append([feasible_actions[i][j][0] + carries[i], feasible_actions[i][j][1] + carries[i]])
        a_merge = np.array(a_merge)

        actions_embed = torch.cat([node_embed[a_merge[:, 0]], node_embed[a_merge[:, 1]]], dim=-1)
        context_embed = graph_embed.repeat_interleave(repeats=torch.tensor(np.array(action_count, dtype=np.int64)).to(actions_embed.device), dim=0)
        actions_embed_with_context = torch.cat([actions_embed, context_embed], dim=-1)

        # forward policy
        action_score = actions_embed_with_context
        for layer in range(self.policy_l):
            action_score = self.policy[layer](action_score)
        sampled_actions, log_prob, mean_entropy = self.sample_actions(scores=action_score, feasible_actions=feasible_actions)

        return sampled_actions, log_prob.unsqueeze(1), mean_entropy


if __name__ == '__main__':
    from env.env_single import JsspN5
    from torch_geometric.data import Batch

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_j = 3
    n_m = 3
    l = 1
    h = 99
    transition = 20
    par = False
    n_workers = 5
    hid_dim = 4

    env = JsspN5(n_job=n_j, n_mch=n_m, low=l, high=h, min_max=True, transition=transition)
    embedding = GIN(in_dim=3, hidden_dim=hid_dim, layer_gin=3).to(dev)
    actor = Actor(3, hid_dim, gin_l=3, policy_l=3).to(dev)

    # t1 = time.time()
    init_s, feasible_a, _ = env.reset()

    scores = embedding(Batch.from_data_list([init_s,init_s]).to(dev))
    actor(Batch.from_data_list([init_s,init_s]).to(dev), [[[8,3],[8,3]],feasible_a])
    # actor(Batch.from_data_list([init_s]).to(dev), [feasible_a])







