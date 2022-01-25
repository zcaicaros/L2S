from __future__ import print_function
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import networkx as nx

import collections
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model


def MinimalJobshopSat(data):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = data
    n_j = len(jobs_data)
    n_m = len(jobs_data[0])

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        machine_assign_mat = []
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                machine_assign_mat.append(assigned_task.job)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        return solver.ObjectiveValue(), np.array(machine_assign_mat).reshape((n_m, n_j))


class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(ForwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(BackwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class Evaluator:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        self.backward_pass = BackwardPass(aggr='max', flow="target_to_source")

    def forward(self, edge_index, duration, n_j, n_m):
        """
        support batch version
        edge_index: [2, n_edges] tensor
        duration: [n_nodes, 1] tensor
        """
        n_nodes = duration.shape[0]
        n_nodes_each_graph = n_j * n_m + 2
        device = edge_index.device

        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        mask_earliest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = duration + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = self.forward_pass(x=x_forward, edge_index=edge_index)
            mask_earliest_start_time = self.forward_pass(x=mask_earliest_start_time, edge_index=edge_index)

        # backward pass...
        index_T = np.cumsum(np.ones(shape=[n_nodes // n_nodes_each_graph], dtype=int) * n_nodes_each_graph) - 1
        make_span = earliest_start_time[index_T]
        # latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        latest_start_time = - torch.ones_like(duration, dtype=torch.float32, device=device)
        latest_start_time[index_T] = - make_span
        mask_latest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_latest_start_time[index_T] = 0
        for _ in range(n_nodes):
            if mask_latest_start_time.sum() == 0:
                break
            x_backward = latest_start_time.masked_fill(mask_latest_start_time.bool(), 0)
            latest_start_time = self.backward_pass(x=x_backward, edge_index=edge_index) + duration
            latest_start_time[index_T] = - make_span
            mask_latest_start_time = self.backward_pass(x=mask_latest_start_time, edge_index=edge_index)

        return earliest_start_time, torch.abs(latest_start_time), make_span


def processing_order_to_edge_index(order, instance):
    """
    order: [n_m, n_j] a numpy array specifying the processing order on each machine, each row is a machine
    instance: [1, n_j, n_m] an instance as numpy array
    RETURN: edge index: [2, n_j * n_m +2] tensor for the directed disjunctive graph
    """
    dur, mch = instance[0], instance[1]
    n_j, n_m = dur.shape[0], dur.shape[1]
    n_opr = n_j*n_m

    adj = np.eye(n_opr, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
    adj[np.arange(start=0, stop=n_opr, step=1).reshape(n_j, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
    adj = np.pad(adj, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
    adj[[i for i in range(1, n_opr + 2 - 1, n_m)], 0] = 1  # connect S with 1st operation of each job
    adj[-1, [i for i in range(n_m, n_opr + 2 - 1, n_m)]] = 1  # connect last operation of each job to T
    adj = np.transpose(adj)

    # rollout ortools solution
    steps_basedon_sol = []
    for i in range(n_m):
        get_col_position_unsorted = np.argwhere(mch == (i + 1))
        get_col_position_sorted = get_col_position_unsorted[order[i]]
        sol_i = order[i] * n_m + get_col_position_sorted[:, 1]
        steps_basedon_sol.append(sol_i.tolist())

    for operations in steps_basedon_sol:
        for i in range(len(operations) - 1):
            adj[operations[i]+1][operations[i+1]+1] += 1

    return torch.nonzero(torch.from_numpy(adj)).t().contiguous()


def forward_pass(graph, topological_order=None):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float('inf'))
    if topological_order is None:
        topo_order = list(nx.topological_sort(graph))
    else:
        topo_order = topological_order
    earliest_ST[topo_order[0]] = 0.
    for n in topo_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]['weight']:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]['weight']
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def backward_pass(graph, makespan, topological_order=None):
    if topological_order is None:
        reverse_order = list(reversed(list(nx.topological_sort(graph))))
    else:
        reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float('inf'))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]['weight']:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]['weight']
    return latest_ST


def forward_and_backward_pass(G):
    # calculate topological order
    topological_order = list(nx.topological_sort(G))
    # forward and backward pass
    est = np.fromiter(forward_pass(graph=G, topological_order=topological_order).values(), dtype=np.float32)
    lst = np.fromiter(backward_pass(graph=G, topological_order=topological_order, makespan=est[-1]).values(), dtype=np.float32)
    # assert np.where(est > lst)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return est, lst, est[-1]


def CPM_batch_G(Gs, dev):
    multi_est = []
    multi_lst = []
    multi_makespan = []
    for G in Gs:
        est, lst, makespan = forward_and_backward_pass(G)
        multi_est.append(est)
        multi_lst.append(lst)
        multi_makespan.append([makespan])
    multi_est = torch.from_numpy(np.concatenate(multi_est, axis=0)).view(-1, 1).to(dev)
    multi_lst = torch.from_numpy(np.concatenate(multi_lst, axis=0)).view(-1, 1).to(dev)
    multi_makespan = torch.tensor(multi_makespan, device=dev)
    return multi_est, multi_lst, multi_makespan



if __name__ == "__main__":
    from generateJSP import uni_instance_gen

    '''j = 10
    m = 10
    l = 1
    h = 99
    batch_size = 1000
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    insts = [np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)]) for _ in range(batch_size)]

    edge_idx_batch = []
    dur_batch = []
    ortools_makespan = []
    eva = Evaluator()
    for i, inst in enumerate(insts):
        print('Processing instance:', i+1)
        times_rearrange = np.expand_dims(inst[0], axis=-1)
        machines_rearrange = np.expand_dims(inst[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        val, sol = MinimalJobshopSat(data.tolist())
        edg_idx = processing_order_to_edge_index(order=sol, instance=inst) + i*(j*m+2)
        edge_idx_batch.append(edg_idx)
        dur = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
        dur_batch.append(dur)
        ortools_makespan.append(val)
    edge_idx_batch = torch.cat(edge_idx_batch, dim=-1)
    dur_batch = torch.cat(dur_batch, dim=0)
    _, _, makespan = eva.forward(edge_index=edge_idx_batch.to(dev), duration=dur_batch.to(dev), n_j=j, n_m=m)
    # print(makespan.squeeze().cpu().numpy())
    # print(ortools_makespan)
    if np.array_equal(makespan.squeeze().cpu().numpy(), np.array(ortools_makespan)):
        print('message-passing evaluator get the same makespan when it rollouts ortools solution.')
    else:
        print('message-passing evaluator get the different makespan when it rollouts ortools solution.')'''


    # start comparing with CPM
    from environment import JsspN5
    from torch_geometric.utils import from_networkx
    from torch_geometric.data.batch import Batch
    j = 100
    m = 20
    l = 1
    h = 99
    batch_size = 1
    dev = 'cuda'
    np.random.seed(1)
    duplicate_one_instance = True

    if not duplicate_one_instance:

        print('Test different instances.')

        insts = np.array([np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)]) for _ in range(batch_size)])

        env = JsspN5(n_job=j, n_mch=m, low=l, high=h, reward_type='yaoxin', fea_norm_const=1)
        states, _, _ = env.reset(instances=insts, init_type='fdd-divide-mwkr', device=dev)

        nx_Gs = env.current_graphs

        # rollout CPM
        CPM_cmax = []
        CPM_schedule = []
        t1 = time.time()
        for G in nx_Gs:
            est = forward_pass(G)
            CPM_cmax.append(max(est.values()))
            CPM_schedule.append(np.fromiter(est.values(), dtype=float))
        t2 = time.time()
        print('CPM takes {} seconds to rollout {} {}x{} instances'.format(t2 - t1, batch_size, j, m))
        CPM_cmax = np.array(CPM_cmax)
        CPM_schedule = np.float32(CPM_schedule)

        if np.array_equal(CPM_cmax, env.incumbent_objs.cpu().numpy().reshape(-1)):
            print('Env reset cmax == CPM cmax ! Congratulations!')

        if np.array_equal(CPM_schedule, np.float32(states[0].cpu().numpy().astype(dtype=float)[:, 1].reshape(batch_size, -1) * env.fea_norm_const)):
            print('Env reset schedule == CPM schedule ! Congratulations!')

        pyg_states = []
        for i in range(batch_size):
            dur = torch.from_numpy(np.pad(insts[i][0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
            pyg_state = from_networkx(nx_Gs[i])
            pyg_state.x = dur
            pyg_states.append(pyg_state)
        pyg_states = Batch.from_data_list(pyg_states).to(dev)

        # rollout message-passing evaluator
        print('Using {}'.format(dev))
        forward_passer = ForwardPass()
        t3 = time.time()
        n_nodes = pyg_states.x.shape[0]
        n_nodes_each_graph = j * m + 2
        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(pyg_states.x, dtype=torch.float32, device=dev)
        mask_earliest_start_time = torch.ones_like(pyg_states.x, dtype=torch.int8, device=dev)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = pyg_states.x + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = forward_passer(x=x_forward, edge_index=pyg_states.edge_index)
            mask_earliest_start_time = forward_passer(x=mask_earliest_start_time, edge_index=pyg_states.edge_index)
        t4 = time.time()
        print('Message-passing takes {} seconds to rollout {} {}x{} instances'.format(t4 - t3, batch_size, j, m))

    else:

        print('Test duplicated instances.')

        insts = np.array([np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)]) for _ in range(1)])

        env = JsspN5(n_job=j, n_mch=m, low=l, high=h, reward_type='yaoxin', fea_norm_const=1)
        states, _, _ = env.reset(instances=insts, init_type='fdd-divide-mwkr', device=dev)

        nx_G = env.current_graphs[0]

        # rollout CPM
        CPM_cmax = []
        CPM_schedule = []
        t1 = time.time()
        for _ in range(batch_size):
            est = forward_pass(nx_G)
            CPM_cmax.append(max(est.values()))
            CPM_schedule.append(np.fromiter(est.values(), dtype=float))
        t2 = time.time()
        print('CPM takes {} seconds to rollout 1 {}x{} instance for {} times.'.format(t2 - t1, j, m, batch_size))
        CPM_cmax = np.array(CPM_cmax)
        CPM_schedule = np.float32(CPM_schedule)

        if np.array_equal(CPM_cmax, env.incumbent_objs.cpu().numpy().reshape(-1)):
            print('Env reset cmax == CPM cmax ! Congratulations!')

        pyg_states = []
        for i in range(batch_size):
            dur = torch.from_numpy(np.pad(insts[0][0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
            pyg_state = from_networkx(nx_G)
            pyg_state.x = dur
            pyg_states.append(pyg_state)
        pyg_states = Batch.from_data_list(pyg_states).to(dev)

        # rollout message-passing evaluator
        print('Using {}'.format(dev))
        forward_passer = ForwardPass()
        t3 = time.time()
        n_nodes = pyg_states.x.shape[0]
        n_nodes_each_graph = j * m + 2
        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(pyg_states.x, dtype=torch.float32, device=dev)
        mask_earliest_start_time = torch.ones_like(pyg_states.x, dtype=torch.int8, device=dev)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = pyg_states.x + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = forward_passer(x=x_forward, edge_index=pyg_states.edge_index)
            mask_earliest_start_time = forward_passer(x=mask_earliest_start_time, edge_index=pyg_states.edge_index)
        t4 = time.time()
        print('Message-passing takes {} seconds to rollout 1 {}x{} instance for {} times.'.format(t4 - t3, j, m, batch_size))


