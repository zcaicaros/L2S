from itertools import count
import numpy as np
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import torch
import random

random.seed(1)
np.random.seed(3)  # 123456324

nn_layer1 = Sequential(Linear(3, 128), ReLU(), Linear(128, 128))
layer1 = GINConv(nn_layer1, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer2 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer2 = GINConv(nn_layer2, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer3 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer3 = GINConv(nn_layer3, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer4 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer4 = GINConv(nn_layer4, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

'''x1 = torch.tensor([[0.0000, 0.0000, 0.0000],
                   [0.1010, 0.0000, 0.0000],
                   [0.7172, 0.0100, 0.0100],
                   [0.8788, 0.2410, 0.2410],
                   [0.7172, 0.0810, 0.0810],
                   [0.2020, 0.1520, 0.1520],
                   [0.5758, 0.1720, 0.2710],
                   [0.8384, 0.0100, 0.0870],
                   [0.0202, 0.1520, 0.1700],
                   [0.6970, 0.1720, 0.1720],
                   [0.0000, 0.3280, 0.3280]])

edge_index1 = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10],
                            [0, 1, 4, 7, 1, 2, 7, 2, 3, 4, 3, 10, 4, 5, 8, 5, 6, 9, 6, 10, 6, 7, 8, 8, 9, 3, 9, 10,
                             10]])

x2 = torch.tensor([[0.0000, 0.0000, 0.0000],
                   [0.2525, 0.0000, 0.0000],
                   [0.0404, 0.0250, 0.0250],
                   [0.5758, 0.0980, 0.2140],
                   [0.7374, 0.2960, 0.3230],
                   [0.0101, 0.3960, 0.3960],
                   [0.2222, 0.4040, 0.4790],
                   [0.2020, 0.4260, 0.5010],
                   [0.7576, 0.4460, 0.5210],
                   [0.4242, 0.6370, 0.8010],
                   [0.1111, 0.6790, 0.8430]])

edge_index2 = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10],
                            [0, 1, 4, 7, 1, 2, 7, 2, 3, 4, 3, 10, 4, 5, 8, 5, 6, 9, 6, 10, 6, 7, 8, 8, 9, 3, 9, 10,
                             10]])



x = torch.cat([x1, x2], dim=0)
edge_index = torch.cat([edge_index1, edge_index2 + 11], dim=-1)
output1 = layer1(x, edge_index)
output2 = layer2(output1, edge_index)
output3 = layer3(output2, edge_index)
output4 = layer4(output3, edge_index)
# print(output4)


x = x1
edge_index = edge_index1
output1 = layer1(x, edge_index)
output2 = layer2(output1, edge_index)
output3 = layer3(output2, edge_index)
output4 = layer4(output3, edge_index)
# print(output4)


x = x2
edge_index = edge_index2
output1 = layer1(x, edge_index)
output2 = layer2(output1, edge_index)
output3 = layer3(output2, edge_index)
output4 = layer4(output3, edge_index)
# print(output4)'''


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index = None
        self.batch = None

    def wrap(self, x, edge_index, batch):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch


batch_data = BatchGraph()

'''from model.actor_v2 import GIN
torch.manual_seed(1)
embedding1 = GIN(in_dim=3, hidden_dim=64, layer_gin=4)'''

from model.actor import GIN

torch.manual_seed(1)
embedding2 = GIN(in_dim=3, hidden_dim=64, layer_gin=4)
# print([param for param in embedding.parameters()])




# print(np.random.randint(low=1, high=4))

# print(torch.isnan(torch.tensor([1, float('nan'), 2])))
# print(torch.tensor([1, float('nan'), 2]).sum().item())
# print(torch.isnan(torch.tensor([1, float('nan'), 2])).sum())

from model.actor import GIN
from env.env_batch import BatchGraph
import torch.optim as optim

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# grad_log = [torch.isnan(param.grad).sum() for param in embedding.parameters()]
# print(torch.stack(grad_log))


# print([1, 2, 3][:-1])

eps = np.finfo(np.float32).eps.item()
tnsr = torch.tensor([666], dtype=torch.float, device=dev)
out = (tnsr - tnsr.mean()) / (torch.std(tnsr, unbiased=False) + eps)
# print(out)
# print((torch.std(tnsr, unbiased=False) + eps))

p_j = 100
p_m = 20
tai_sota = np.array([5464, 5181, 5568, 5339, 5392, 5342, 5436, 5394, 5358, 5183], dtype=float)
np.save('./test_data/tai{}x{}_SOTA_result.npy'.format(p_j, p_m), tai_sota)

from parameters import args

init = args.init_type
print('./saved_model/{}x{}[{},{}]_{}_{}_{}_'  # env parameters
      '{}_{}_{}_'  # model parameters
      '{}_{}_{}_{}_{}_{}_'  # training parameters
      'incumbent.pth'  # saving model type
      .format(args.j, args.m, args.l, args.h, init, args.reward_type, args.gamma,
              args.hidden_dim, args.embedding_layer, args.policy_layer,
              args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation))

print('./saved_model/{}x{}[{},{}]_{}_{}_{}_'  # env parameters
      '{}_{}_{}_'  # model parameters
      '{}_{}_{}_{}_{}_{}_'  # training parameters
      'last-step.pth'  # saving model type
      .format(args.j, args.m, args.l, args.h, init, args.reward_type, args.gamma,
              args.hidden_dim, args.embedding_layer, args.policy_layer,
              args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
              args.step_validation))

print('./log/training_log_'
      '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
      '{}_{}_{}_'  # model parameters
      '{}_{}_{}_{}_{}_{}.npy'  # training parameters
      .format(args.j, args.m, args.l, args.h, init, args.reward_type, args.gamma,
              args.hidden_dim, args.embedding_layer, args.policy_layer,
              args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
              args.step_validation))

print('./log/validation_log_'
      '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
      '{}_{}_{}_'  # model parameters
      '{}_{}_{}_{}_{}_{}.npy'  # training parameters
      .format(args.j, args.m, args.l, args.h, init, args.reward_type, args.gamma,
              args.hidden_dim, args.embedding_layer, args.policy_layer,
              args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
              args.step_validation))
