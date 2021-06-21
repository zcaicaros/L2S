import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import env.jsp_problem as jsp
from torch_geometric.data import Data


def plot_sol(solution, n_job, n_mch):
    x_axis = np.pad(np.tile(np.arange(1, n_mch + 1, 1), n_job), (1, 1), 'constant',
                    constant_values=[0, n_mch + 1])
    y_axis = np.pad(np.arange(n_job, 0, -1).repeat(n_mch), (1, 1), 'constant',
                    constant_values=np.median(np.arange(n_job, 0, -1)))
    pos = dict((n, (x, y)) for n, x, y in zip(solution.nodes(), x_axis, y_axis))
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    nx.draw_networkx_edge_labels(solution, pos=pos)  # show edge weight
    nx.draw(
        solution, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'  # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
    )
    plt.tight_layout()
    plt.show()


def dag2pyg(**kwargs):
    G = kwargs['G']
    instance, high = kwargs['instance'], kwargs['high']
    n_job, n_mch = instance[0].shape[0], instance[0].shape[1]
    n_oprs = n_job * n_mch
    min_max, normalizer = kwargs['min_max'], kwargs['normalizer']

    # start to build PyG data
    adj = nx.to_numpy_matrix(G)
    adj[0, [i for i in range(1, n_oprs + 2 - 1, n_mch)]] = 1
    np.fill_diagonal(adj, 1)
    topological_order = list(nx.topological_sort(G))
    est_ST = np.fromiter(jsp.forward_pass(graph=G, topological_order=topological_order).values(), dtype=np.float32)
    lst_ST = np.fromiter(
        jsp.backward_pass(graph=G, topological_order=topological_order, makespan=est_ST[-1]).values(), dtype=np.float32)
    # print(est_ST)
    # print(lst_ST)
    f1 = torch.from_numpy(
        np.pad(np.float32((instance[0].reshape(-1, 1)) / high), ((1, 1), (0, 0)), 'constant', constant_values=0))
    if min_max:
        normalizer.fit(est_ST.reshape(-1, 1))
        f2 = torch.from_numpy(normalizer.transform(est_ST.reshape(-1, 1)))
        normalizer.fit(lst_ST.reshape(-1, 1))
        f3 = torch.from_numpy(normalizer.transform(lst_ST.reshape(-1, 1)))
    else:
        f2 = torch.from_numpy(est_ST.reshape(-1, 1)/1000)
        f3 = torch.from_numpy(lst_ST.reshape(-1, 1)/1000)
    x = torch.cat([f1, f2, f3], dim=-1)
    edge_idx = torch.nonzero(torch.from_numpy(adj)).t().contiguous()
    # print(adj)
    # print(edge_idx)
    return Data(x=x, edge_index=edge_idx, y=np.amax(est_ST))
