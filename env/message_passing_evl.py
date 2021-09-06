from __future__ import print_function
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

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
        latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
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



if __name__ == "__main__":
    from generateJSP import uni_instance_gen
    from env.env_single import JsspN5
    import time
    from torch_geometric.data.batch import Batch

    j = 15
    m = 15
    l = 1
    h = 99
    batch_size = 10
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, init='rule', rule='fdd-divide-mwkr', transition=0)
    insts = [np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)]) for _ in range(batch_size)]

    '''for _ in range(1, batch_size):
        state, feasible_action, done = env.reset(instance=insts[_], fix_instance=True)'''

    for inst in insts:

        # test networkx forward and backward pass...
        t1 = time.time()
        state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
        t2 = time.time()

        # testing forward pass...
        dur_earliest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1).to(dev)
        forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        earliest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32, device=dev)
        adj_earliest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]].to(dev)
        ma_earliest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8, device=dev)
        ma_earliest_st[0] = 0

        t3 = time.time()
        for _ in range(j*m+2):
            if ma_earliest_st.sum() == 0:
                print('finish forward pass at step:', _)
                break
            x = dur_earliest_st + earliest_st.masked_fill(ma_earliest_st.bool(), 0)
            earliest_st = forward_pass(x=x, edge_index=adj_earliest_st)
            ma_earliest_st = forward_pass(x=ma_earliest_st, edge_index=adj_earliest_st)
        t4 = time.time()
        if torch.equal(earliest_st.cpu().squeeze() / 1000, state.x[:, 1]):
            print('forward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)

        # testing backward pass...
        state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
        dur_latest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1).to(dev)
        backward_pass = BackwardPass(aggr='max', flow="target_to_source")
        latest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32, device=dev)
        latest_st[-1] = - float(state.y)
        adj_latest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]].to(dev)
        ma_latest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8, device=dev)
        ma_latest_st[-1] = 0
        t3 = time.time()
        for _ in range(j * m + 2):  # j * m + 2
            if ma_latest_st.sum() == 0:
                print('finish backward pass at step:', _)
                break
            x = latest_st.masked_fill(ma_latest_st.bool(), 0)
            latest_st = backward_pass(x=x, edge_index=adj_latest_st) + dur_latest_st
            latest_st[-1] = - float(state.y)
            ma_latest_st = backward_pass(x=ma_latest_st, edge_index=adj_latest_st)
        t4 = time.time()
        if torch.equal(- latest_st.squeeze().cpu() / 1000, state.x[:, 2]):
            print('backward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)

        # test hybrid evaluator
        state_list = []
        dur_list = []
        for _ in range(batch_size):
            state, feasible_action, done = env.reset(instance=insts[_], fix_instance=True)
            state_list.append(state)
            dur_list.append(np.pad(insts[_][0].reshape(-1), (1, 1), 'constant', constant_values=0))
        batch_data = Batch.from_data_list(state_list)
        edge_idx = batch_data.edge_index[:, batch_data.edge_index[0] != batch_data.edge_index[1]].to(dev)
        dur = np.concatenate(dur_list)
        dur = torch.from_numpy(dur).reshape(-1, 1).to(dev)
        eva = Evaluator()
        t5 = time.time()
        est, lst, makespan = eva.forward(edge_index=edge_idx, duration=dur, n_j=j, n_m=m)
        t6 = time.time()
        # print(makespan)
        # print(est.cpu().reshape(batch_size, -1, 1).max(dim=1))
        # print(lst.cpu().reshape(batch_size, -1, 1).max(dim=1))
        if torch.equal(est.cpu().squeeze() / 1000, batch_data.x[:, 1]) and torch.equal(lst.squeeze().cpu() / 1000, batch_data.x[:, 2]):
            print('forward pass and backward pass are all OK! It takes:', t6 - t5, 'networkx version forward pass and backward pass take:', t2 - t1)

        # get ortools solution...
        times_rearrange = np.expand_dims(inst[0], axis=-1)
        machines_rearrange = np.expand_dims(inst[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        val, sol = MinimalJobshopSat(data.tolist())
        edg_idx = processing_order_to_edge_index(order=sol, instance=inst)
        eva_for_ortools = Evaluator()
        dur = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
        _, _, makespan = eva.forward(edge_index=edg_idx, duration=dur, n_j=j, n_m=m)
        print('makespan of message-passing evaluator:', makespan.cpu().numpy()[0][0])
        print('makespan of ortools:', val)

        print()

