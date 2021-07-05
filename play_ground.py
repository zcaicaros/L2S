from itertools import count
import numpy as np
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import torch


nn_layer1 = Sequential(Linear(3, 128), ReLU(), Linear(128, 128))
layer1 = GINConv(nn_layer1, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer2 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer2 = GINConv(nn_layer2, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer3 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer3 = GINConv(nn_layer3, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

nn_layer4 = Sequential(Linear(128, 128), ReLU(), Linear(128, 128))
layer4 = GINConv(nn_layer4, eps=0, train_eps=False, aggr='mean', flow="source_to_target")

x = torch.tensor([[0.0000, 0.0000, 0.0000],
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

edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10],
                           [0, 1, 4, 7, 1, 2, 7, 2, 3, 4, 3, 10, 4, 5, 8, 5, 6, 9, 6, 10, 6, 7, 8, 8, 9, 3, 9, 10, 10]])

output1 = layer1(x, edge_index)
output2 = layer2(output1, edge_index)
output3 = layer3(output2, edge_index)
output4 = layer4(output3, edge_index)
print(output4)

print(torch.std(torch.tensor([1.])))

print(np.load('./log/validation_log_10x10_6.4w_spt.npy'))
