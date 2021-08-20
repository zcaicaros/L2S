from env.env_batch import JsspN5
import torch
import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from env.message_passing_evl import Evaluator
from torch_geometric.utils import from_networkx
from torch_geometric.data.batch import Batch


def show_state(G, j, m):
    x_axis = np.pad(np.tile(np.arange(1, m + 1, 1), j), (1, 1), 'constant', constant_values=[0, m + 1])
    y_axis = np.pad(np.arange(j, 0, -1).repeat(m), (1, 1), 'constant', constant_values=np.median(np.arange(j, 0, -1)))
    pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
    nx.draw(
        G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'
        # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
    )
    plt.show()


def change_nxgraph_topology(action, G, instance, plot=False):
    """
    action: list e.g. [0, 0]
    G: networkx Digraph object representing a solution
    instance: corresponding np.array instance [2, j, m]
    """
    n_jobs, n_machines = instance[0].shape
    n_operations = n_jobs * n_machines
    G_copy = copy.deepcopy(G)

    if action == [0, 0]:  # if dummy action then do nothing
        pass
    else:  # change nx graph topology
        S = [s for s in G_copy.predecessors(action[0]) if
             int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
        T = [t for t in G_copy.successors(action[1]) if
             int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
        s = S[0] if len(S) != 0 else None
        t = T[0] if len(T) != 0 else None

        if s is not None:  # connect s with action[1]
            G_copy.remove_edge(s, action[0])
            G_copy.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
        else:
            pass

        if t is not None:  # connect action[0] with t
            G_copy.remove_edge(action[1], t)
            G_copy.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
        else:
            pass

        # reverse edge connecting selected pair
        G_copy.remove_edge(action[0], action[1])
        G_copy.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

    if plot:
        show_state(G_copy, n_jobs, n_machines)

    return G_copy


def get_initial_sols(instances, low, high, init_type, dev):
    """
    instances: np.array [batch_size, 2, j, m]
    low: lower bound for processing time of instances
    high: higher bound for processing time of instances
    init_type: 'fdd-divide-mwkr', 'spt', or 'plist'
    """
    j, m = instances[0][0].shape
    env = JsspN5(j, m, low, high)
    env.reset(instances=instances, init_type=init_type, device=dev)
    init_Gs = copy.deepcopy(env.current_graphs)
    return init_Gs  # list of networkx.Digraph


def _get_pairs(cb, cb_op, tabu_list=None):
    pairs = []
    rg = cb[:-1].shape[0]  # sliding window of 2
    for i in range(rg):
        if cb[i] == cb[i + 1]:  # find potential pair
            if i == 0:
                if cb[i + 1] != cb[i + 2]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
            elif cb[i] != cb[i - 1]:
                if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                    pairs.append([cb_op[i], cb_op[i + 1]])
            elif i + 1 == rg:
                if cb[i + 1] != cb[i]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
            elif cb[i + 1] != cb[i + 2]:
                if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                    pairs.append([cb_op[i], cb_op[i + 1]])
            else:
                pass
    return pairs


def _gen_moves(G, mch_mat, tabu_list):
    """
    G: networkx Digraph object representing a solution
    mch_mat: np.array [j, m]
    tabu_list: list of forbidden moves, e.g. [[3,5], [5,10], ...] for G
    """
    critical_path = nx.dag_longest_path(G)[1:-1]
    critical_blocks_opr = np.array(critical_path)
    critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 0
    pairs = _get_pairs(critical_blocks, critical_blocks_opr, tabu_list)
    return pairs


def feasible_action(G, tabu_list, instance):
    """
    G: networkx Digraph object representing a solution
    tabu_list: list of forbidden moves, e.g. [[3,5], [5,10], ...] for G
    instance: corresponding np.array instance [2, j, m]
    """
    action = _gen_moves(G=G, mch_mat=instance[1], tabu_list=tabu_list)
    if len(action) != 0:
        return action
    else:  # if no feasible actions available return [0, 0]
        return [0, 0]


def random_policy_baselines(instances, search_horizon, dev, init_type='fdd-divide-mwkr', low=1, high=99):
    """
    instances: np.array [batch_size, 2, j, m]
    """
    j, m = instances[0][0].shape
    n_op = j * m
    horizon = 0
    eva = Evaluator()
    tabu_size = 1
    tabu_lst = [[] for _ in range(instances.shape[0])]

    x = torch.from_numpy(np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0).reshape(-1, 1))

    Gs = get_initial_sols(instances=instances, low=low, high=high, init_type=init_type, dev=dev)
    pyg = Batch.from_data_list([from_networkx(G) for G in Gs])
    _, _, make_span = eva.forward(pyg.edge_index.to(dev), duration=x.to(dev), n_j=j, n_m=m)

    incumbent_makespan = make_span
    while horizon < search_horizon:
        feasible_actions = [feasible_action(G, tl, ins) for G, tl, ins in zip(Gs, tabu_lst, instances)]
        actions = [random.choice(feasible_a) for feasible_a in feasible_actions]  # [[1, 2], [5, 8], ...]
        # print(actions[0][::-1])
        print([a for a in actions])
        action_reversed = [a[::-1] for a in actions]
        for i, action in enumerate(action_reversed):
            if action == [0, 0]:  # if dummy action, don't update tabu list
                pass
            else:
                if len(tabu_lst[i]) == tabu_size:
                    tabu_lst[i].pop(0)
                    tabu_lst[i].append(action)
                else:
                    tabu_lst[i].append(action)
        Gs = [change_nxgraph_topology(a, G, ins) for a, G, ins in zip(actions, Gs, instances)]
        pyg = Batch.from_data_list([from_networkx(G) for G in Gs])
        _, _, make_span = eva.forward(pyg.edge_index.to(dev), duration=x.to(dev), n_j=j, n_m=m)
        incumbent_makespan = torch.where(make_span - incumbent_makespan < 0, make_span, incumbent_makespan)
        horizon += 1









if __name__ == "__main__":
    from env.generateJSP import uni_instance_gen

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    problem = 'tai'
    j = 4
    m = 4
    l = 1
    h = 99
    b = 10
    init = 'fdd-divide-mwkr'
    steps = 10

    random.seed(3)
    np.random.seed(2)

    # insts = np.load('./test_data/{}{}x{}.npy'.format(problem, j, m))
    insts = np.array([uni_instance_gen(j, m, l, h) for _ in range(b)])

    init_sols = get_initial_sols(instances=insts, low=l, high=h, init_type=init, dev=device)

    sol = init_sols[1]
    # show_state(sol, j, m)
    tabu_l = []
    candidate_action = feasible_action(sol, tabu_l, insts[1])

    random_policy_baselines(instances=insts, search_horizon=steps, dev=device)

