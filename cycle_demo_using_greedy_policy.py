from env.environment import JsspN5
import time
import torch
import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from env.message_passing_evl import Evaluator
from torch_geometric.utils import from_networkx
from torch_geometric.data.batch import Batch
from ortools_solver import MinimalJobshopSat


class LongTermMem:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.mem = []

    def add_ele(self, element):
        if len(self.mem) < self.mem_size:
            self.mem.append(element)
        else:
            self.mem.pop(0)
            self.mem.append(element)

    def sample_ele(self):
        return random.choice(self.mem)

    def clean_mem(self):
        self.mem = []


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
        return [[0, 0]]


def Greedy_baselines(instances, search_horizon, log_step, dev, init_type='fdd-divide-mwkr', low=1, high=99):
    """
    instances: np.array [batch_size, 2, j, m]
    search_horizon: int
    """

    time_start = time.time()
    time_log = []

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

    results = []
    incumbent_makespan = make_span
    while horizon < search_horizon:
        feasible_actions = [feasible_action(G, tl, ins) for G, tl, ins in zip(Gs, tabu_lst, instances)]

        # start to find move for all instances...
        next_state_count = [len(fea_a) for fea_a in feasible_actions]
        dur_for_find_move = np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0)
        dur_for_find_move = np.repeat(dur_for_find_move, next_state_count, axis=0).reshape(-1, 1)
        dur_for_find_move = torch.from_numpy(dur_for_find_move)
        Gs_all_instances = []
        for fea_a, G, ins in zip(feasible_actions, Gs, instances):  # fea_a: e.g. [[1, 2], [6, 8], ...]
            for a in fea_a:  # a: e.g. [1, 2]
                Gs_all_instances.append(change_nxgraph_topology(a, G, ins))
        pyg_one_step_fwd = Batch.from_data_list([from_networkx(G) for G in Gs_all_instances])
        _, _, make_span = eva.forward(pyg_one_step_fwd.edge_index.to(dev), duration=dur_for_find_move.to(dev), n_j=j, n_m=m)
        make_span = make_span.cpu().numpy()
        actions_idx = [np.argmin(make_span[start:end]) for start, end in zip(np.cumsum([0]+next_state_count[:-1]), np.cumsum(next_state_count))]
        selected_actions = [fea_a[idx] for fea_a, idx in zip(feasible_actions, actions_idx)]

        print(selected_actions)

        # move...
        action_reversed = [a[::-1] for a in selected_actions]
        for i, action in enumerate(action_reversed):
            if action == [0, 0]:  # if dummy action, don't update tabu list
                pass
            else:
                if len(tabu_lst[i]) == tabu_size:
                    tabu_lst[i].pop(0)
                    tabu_lst[i].append(action)
                else:
                    tabu_lst[i].append(action)
        Gs = [change_nxgraph_topology(a, G, ins) for a, G, ins in zip(selected_actions, Gs, instances)]
        pyg = Batch.from_data_list([from_networkx(G) for G in Gs])
        _, _, make_span = eva.forward(pyg.edge_index.to(dev), duration=x.to(dev), n_j=j, n_m=m)
        incumbent_makespan = torch.where(make_span - incumbent_makespan < 0, make_span, incumbent_makespan)
        horizon += 1

        for log_t in log_step:
            if horizon == log_t:
                results.append(incumbent_makespan.cpu().numpy().reshape(-1))
                time_log.append((time.time() - time_start) / instances.shape[0])

    return np.stack(results), np.array(time_log)


def BestImprovement_baseline(instances, search_horizon, log_step, dev, init_type='fdd-divide-mwkr', low=1, high=99):
    """
    instances: np.array [batch_size, 2, j, m]
    search_horizon: int
    """

    time_start = time.time()
    time_log = []

    j, m = instances[0][0].shape
    n_op = j * m
    horizon = 0
    eva = Evaluator()
    tabu_size = 1
    tabu_lst = [[] for _ in range(instances.shape[0])]
    batch_memory = [LongTermMem(mem_size=100) for _ in range(instances.shape[0])]

    dur_for_move = torch.from_numpy(np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0).reshape(-1, 1))

    current_Gs = get_initial_sols(instances=instances, low=low, high=high, init_type=init_type, dev=dev)
    current_pyg = Batch.from_data_list([from_networkx(G) for G in current_Gs])
    _, _, init_make_span = eva.forward(current_pyg.edge_index.to(dev), duration=dur_for_move.to(dev), n_j=j, n_m=m)

    results = []
    incumbent_makespan = init_make_span
    # print(incumbent_makespan.squeeze())
    while horizon < search_horizon:
        feasible_actions = [feasible_action(G, tl, ins) for G, tl, ins in zip(current_Gs, tabu_lst, instances)]

        ## start to find move for all instances...
        # calculate next G of all actions for all instances
        Gs_for_find_move = [[] for _ in range(len(feasible_actions))]
        actions_for_find_move = [[] for _ in range(len(feasible_actions))]
        next_G_count = [len(fea_a) for fea_a in feasible_actions]
        for i, (fea_a, G, t_l, ins) in enumerate(zip(feasible_actions, current_Gs, tabu_lst, instances)):  # fea_a: e.g. [[1, 2], [6, 8], ...]
            for a in fea_a:  # a: e.g. [1, 2]
                actions_for_find_move[i].append(a)
                if a != [0, 0]:
                    Gs_for_find_move[i].append(change_nxgraph_topology(a, G, ins))
                    if len(t_l) == tabu_size:
                        t_l.pop(0)
                        t_l.append(a[::-1])
                    else:
                        t_l.append(a[::-1])
                    batch_memory[i].add_ele([change_nxgraph_topology(a, G, ins), t_l])
                else:
                    Gs_for_find_move[i].append(change_nxgraph_topology(a, G, ins))
        # batching all next G
        pyg_one_step_fwd = Batch.from_data_list([from_networkx(G) for i in range(len(feasible_actions)) for G in Gs_for_find_move[i]])
        # calculate dur for evaluator
        dur_for_find_move = np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0)
        dur_for_find_move = np.repeat(dur_for_find_move, next_G_count, axis=0).reshape(-1, 1)
        dur_for_find_move = torch.from_numpy(dur_for_find_move)
        # calculate make_span for all next G of all instances
        _, _, make_span_for_find_moves = eva.forward(pyg_one_step_fwd.edge_index.to(dev), duration=dur_for_find_move.to(dev), n_j=j, n_m=m)
        make_span_for_find_moves = make_span_for_find_moves.cpu().numpy()
        min_make_span_idx_for_find_moves = [np.argmin(make_span_for_find_moves[start:end]) for start, end in zip(np.cumsum([0]+next_G_count[:-1]), np.cumsum(next_G_count))]
        min_make_span_for_find_moves = [ms[idx][0] for ms, idx in zip([make_span_for_find_moves[start:end] for start, end in zip(np.cumsum([0]+next_G_count[:-1]), np.cumsum(next_G_count))], min_make_span_idx_for_find_moves)]
        flag_need_restart = (incumbent_makespan < torch.tensor(min_make_span_for_find_moves, device=incumbent_makespan.device).reshape(-1, 1)).squeeze().cpu().numpy()
        if flag_need_restart.size == 1:
            flag_need_restart = flag_need_restart.reshape(1)
        for i, (flag, min_idx) in enumerate(zip(flag_need_restart, min_make_span_idx_for_find_moves)):
            if flag:  # random restart from long-term memory
                current_Gs[i], tabu_lst[i] = random.choice(batch_memory[i].mem)
            else:  # move
                if actions_for_find_move[i][min_idx] != [0, 0]:
                    current_Gs[i] = Gs_for_find_move[i][min_idx]
                    if len(tabu_lst[i]) == tabu_size:
                        tabu_lst[i].pop(0)
                        tabu_lst[i].append(actions_for_find_move[i][min_idx][::-1])
                    else:
                        tabu_lst[i].append(actions_for_find_move[i][min_idx][::-1])
                else:
                    current_Gs[i], tabu_lst[i] = Gs_for_find_move[i][min_idx], [tabu_lst[i]]

        current_pyg = Batch.from_data_list([from_networkx(G) for G in current_Gs])
        _, _, make_span = eva.forward(current_pyg.edge_index.to(dev), duration=dur_for_move.to(dev), n_j=j, n_m=m)
        incumbent_makespan = torch.where(make_span - incumbent_makespan < 0, make_span, incumbent_makespan)
        horizon += 1

        for log_t in log_step:
            if horizon == log_t:
                results.append(incumbent_makespan.cpu().numpy().reshape(-1))
                time_log.append((time.time() - time_start) / instances.shape[0])

    return np.stack(results), np.array(time_log)


def FirstImprovement_baseline(instances, search_horizon, log_step, dev, init_type='fdd-divide-mwkr', low=1, high=99):
    """
    instances: np.array [batch_size, 2, j, m]
    search_horizon: int
    """

    time_start = time.time()
    time_log = []

    j, m = instances[0][0].shape
    n_op = j * m
    horizon = 0
    eva = Evaluator()
    tabu_size = 1
    tabu_lst = [[] for _ in range(instances.shape[0])]
    batch_memory = [LongTermMem(mem_size=100) for _ in range(instances.shape[0])]

    dur_for_move = torch.from_numpy(np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0).reshape(-1, 1))

    current_Gs = get_initial_sols(instances=instances, low=low, high=high, init_type=init_type, dev=dev)
    current_pyg = Batch.from_data_list([from_networkx(G) for G in current_Gs])
    _, _, init_make_span = eva.forward(current_pyg.edge_index.to(dev), duration=dur_for_move.to(dev), n_j=j, n_m=m)

    results = []
    incumbent_makespan = init_make_span.cpu().numpy().reshape(-1)
    # print(incumbent_makespan.squeeze())
    while horizon < search_horizon:
        feasible_actions = [feasible_action(G, tl, ins) for G, tl, ins in zip(current_Gs, tabu_lst, instances)]

        ## start to find move for all instances...
        # calculate next G of all actions for all instances
        Gs_for_find_move = [[] for _ in range(len(feasible_actions))]
        actions_for_find_move = [[] for _ in range(len(feasible_actions))]
        next_G_count = [len(fea_a) for fea_a in feasible_actions]
        for i, (fea_a, G, t_l, ins) in enumerate(zip(feasible_actions, current_Gs, tabu_lst, instances)):  # fea_a: e.g. [[1, 2], [6, 8], ...]
            for a in fea_a:  # a: e.g. [1, 2]
                actions_for_find_move[i].append(a)
                if a != [0, 0]:
                    Gs_for_find_move[i].append(change_nxgraph_topology(a, G, ins))
                    if len(t_l) == tabu_size:
                        t_l.pop(0)
                        t_l.append(a[::-1])
                    else:
                        t_l.append(a[::-1])
                    batch_memory[i].add_ele([change_nxgraph_topology(a, G, ins), t_l])
                else:
                    Gs_for_find_move[i].append(change_nxgraph_topology(a, G, ins))
        # batching all next G
        pyg_one_step_fwd = Batch.from_data_list([from_networkx(G) for i in range(len(feasible_actions)) for G in Gs_for_find_move[i]])
        # calculate dur for evaluator
        dur_for_find_move = np.pad(instances[:, 0].reshape(-1, n_op), ((0, 0), (1, 1)), 'constant', constant_values=0)
        dur_for_find_move = np.repeat(dur_for_find_move, next_G_count, axis=0).reshape(-1, 1)
        dur_for_find_move = torch.from_numpy(dur_for_find_move)
        # calculate make_span for all next G of all instances
        _, _, make_span_for_find_moves = eva.forward(pyg_one_step_fwd.edge_index.to(dev), duration=dur_for_find_move.to(dev), n_j=j, n_m=m)
        make_span_for_find_moves = make_span_for_find_moves.cpu().numpy().reshape(-1)
        splited_make_span_for_find_moves = [make_span_for_find_moves[start:end] for start, end in zip(np.cumsum([0]+next_G_count[:-1]), np.cumsum(next_G_count))]
        first_smaller_idx = [np.argmax(ms < target) for ms, target in zip(splited_make_span_for_find_moves, incumbent_makespan)]
        first_smaller_make_span = [ms[idx] for ms, idx in zip(splited_make_span_for_find_moves, first_smaller_idx)]
        flag_need_restart = incumbent_makespan < first_smaller_make_span

        for i, (flag, fst_smaller_idx) in enumerate(zip(flag_need_restart, first_smaller_idx)):
            if flag:  # random restart from long-term memory
                current_Gs[i], tabu_lst[i] = random.choice(batch_memory[i].mem)
            else:  # move
                if actions_for_find_move[i][fst_smaller_idx] != [0, 0]:
                    current_Gs[i] = Gs_for_find_move[i][fst_smaller_idx]
                    if len(tabu_lst[i]) == tabu_size:
                        tabu_lst[i].pop(0)
                        tabu_lst[i].append(actions_for_find_move[i][fst_smaller_idx][::-1])
                    else:
                        tabu_lst[i].append(actions_for_find_move[i][fst_smaller_idx][::-1])
                else:
                    current_Gs[i], tabu_lst[i] = Gs_for_find_move[i][fst_smaller_idx], [tabu_lst[i]]

        current_pyg = Batch.from_data_list([from_networkx(G) for G in current_Gs])
        _, _, make_span = eva.forward(current_pyg.edge_index.to(dev), duration=dur_for_move.to(dev), n_j=j, n_m=m)
        make_span = make_span.cpu().numpy().reshape(-1)
        incumbent_makespan = np.where(make_span - incumbent_makespan < 0, make_span, incumbent_makespan)
        horizon += 1

        for log_t in log_step:
            if horizon == log_t:
                results.append(incumbent_makespan)
                time_log.append((time.time() - time_start) / instances.shape[0])

    return np.stack(results), np.array(time_log)


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # benchmark config
    l = 1
    h = 99
    init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
    testing_type = ['syn']  # ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn']
    syn_problem_j = [10, 15, 15, 20, 20, 100, 150]  # [10, 15, 15, 20, 20, 100, 150]
    syn_problem_m = [10, 10, 15, 10, 15, 20, 25]  # [10, 10, 15, 10, 15, 20, 25]
    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]  # [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]  # [15, 15, 20, 15, 20, 15, 20, 20]
    abz_problem_j = [10, 20]  # [10, 20]
    abz_problem_m = [10, 15]  # [10, 15]
    orb_problem_j = [10]  # [10]
    orb_problem_m = [10]  # [10]
    yn_problem_j = [20]  # [20]
    yn_problem_m = [20]  # [20]
    swv_problem_j = [20, 20, 50]  # [20, 20, 50]
    swv_problem_m = [10, 15, 10]  # [10, 15, 10]
    la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]
    ft_problem_j = [6, 10, 20]  # [6, 10, 20]
    ft_problem_m = [6, 10, 5]  # [6, 10, 5]

    # MDP config
    cap_horizon = 5000  # 5000
    transit = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]

    for test_t in testing_type:  # select benchmark
        if test_t == 'syn':
            problem_j, problem_m = syn_problem_j, syn_problem_m
        elif test_t == 'tai':
            problem_j, problem_m = tai_problem_j, tai_problem_m
        elif test_t == 'abz':
            problem_j, problem_m = abz_problem_j, abz_problem_m
        elif test_t == 'orb':
            problem_j, problem_m = orb_problem_j, orb_problem_m
        elif test_t == 'yn':
            problem_j, problem_m = yn_problem_j, yn_problem_m
        elif test_t == 'swv':
            problem_j, problem_m = swv_problem_j, swv_problem_m
        elif test_t == 'la':
            problem_j, problem_m = la_problem_j, la_problem_m
        elif test_t == 'ft':
            problem_j, problem_m = ft_problem_j, ft_problem_m
        else:
            raise Exception(
                'Problem type must be in testing_type = ["tai", "abz", "orb", "yn", "swv", "la", "ft", "syn"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            testing_instances = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))[:1]
            print('\nStart testing {}{}x{}...\n'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            if test_t != 'syn':
                gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
            else:
                # ortools solver
                from pathlib import Path
                ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if ortools_path.is_file():
                    gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                else:
                    ortools_results = []
                    print('Starting Ortools...')
                    for i, data in enumerate(testing_instances):
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        ortools_results.append(result)
                    ortools_results = np.array(ortools_results)
                    np.save('./test_data/syn{}x{}_result.npy'.format(p_j, p_m), ortools_results)
                    gap_against = ortools_results[:, 1]

            for init in init_type:

                print('Testing Greedy Policy...')
                greedy_makespan, greedy_time = Greedy_baselines(instances=testing_instances, search_horizon=cap_horizon, log_step=transit, dev=dev, init_type=init, low=l, high=h)
                gap_greedy_policy = ((greedy_makespan - gap_against) / gap_against).mean(axis=-1)
                print('Greedy policy gap for {} testing steps are: {}'.format(transit, gap_greedy_policy))
                print('Greedy policy time for {} testing steps are: {}'.format(transit, greedy_time))


if __name__ == "__main__":

    main()