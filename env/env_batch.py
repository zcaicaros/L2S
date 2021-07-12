import os
import sys

import torch_geometric.utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import networkx as nx
from env.generateJSP import uni_instance_gen
import env.jsp_problem as jsp
from sklearn.preprocessing import MinMaxScaler
from env.utils import plot_sol, dag2pyg
from torch_geometric.data import Data
from model.actor import Actor
from env.permissible_LS import permissibleLeftShift
from parameters import args as parameters
from env.jsp_problem import forward_and_backward_pass
from propagate_evl import Evaluator
import matplotlib.pyplot as plt


class JsspN5:
    def __init__(self,
                 n_job,
                 n_mch,
                 low,
                 high,
                 init='p_list',
                 rule='spt',
                 transition=100):

        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        self.init = init
        self.rule = rule
        self.itr = 0
        self.max_transition = transition
        self.instances = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.incumbent_idle = None
        self.fea_norm_const = 1000
        self.eva = Evaluator()

    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        """
        solution: networkx DAG conjunctive graph
        mch_mat: the same mch from our NeurIPS 2020 paper of solution
        """
        critical_path = nx.dag_longest_path(solution)[1:-1]
        critical_blocks_opr = np.array(critical_path)
        critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 0
        pairs = self._get_pairs(critical_blocks, critical_blocks_opr, tabu_list)
        return pairs

    @staticmethod
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

    @staticmethod
    def _get_pairs_has_tabu(cb, cb_op):
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    '''def _p_list_solver_single_instance(self, args, plot=False):
        instance = args[0]
        # print(instance)
        priority_list = args[1]
        # print(priority_list)
        make_span, earliest_start, latest_start, adj_aug, G = jsp.eval_priority_list(p_list=priority_list,
                                                                                     dur_mat=instance[0],
                                                                                     mch_mat=instance[1],
                                                                                     plot=plot)
        earliest_start = earliest_start.astype(np.float32)
        latest_start = latest_start.astype(np.float32)
        f1 = torch.from_numpy(
            np.pad(np.float32((instance[0].reshape(-1, 1)) / self.high), ((1, 1), (0, 0)), 'constant',
                   constant_values=0))
        f2 = torch.from_numpy(earliest_start.reshape(-1, 1) / 1000)
        f3 = torch.from_numpy(latest_start.reshape(-1, 1) / 1000)
        x = torch.cat([f1, f2, f3], dim=-1)
        edge_idx = torch.nonzero(torch.from_numpy(adj_aug)).t().contiguous()
        init_state = Data(x=x, edge_index=edge_idx, y=np.amax(earliest_start))
        return init_state, G'''

    '''def rules_solver(self, instance, plot=False):
        dur_mat, dur_cp, mch_mat = instance[0], np.copy(instance[0]), instance[1]
        n_job, n_mch = dur_mat.shape[0], dur_mat.shape[1]
        n_opr = n_job * n_mch
        last_col = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, -1]
        first_col = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, 0]
        # initialize action space: [n_job, 1], the first column
        candidate_oprs = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, 0]
        # initialize the mask: [n_job, 1]
        mask = np.zeros(shape=n_job, dtype=bool)
        # initialize adj matrix
        conj_nei_up_stream = np.eye(n_opr, k=-1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[first_col] = 0
        adj = conj_nei_up_stream

        gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
        opIDsOnMchs = -n_job * np.ones_like(dur_mat.transpose(), dtype=np.int32)
        finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

        actions = []
        for _ in range(n_opr):

            if self.rule == 'spt':
                candidate_masked = candidate_oprs[np.where(~mask)]
                dur_candidate = np.take(dur_mat, candidate_masked)
                idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                action = candidate_masked[idx]
            elif self.rule == 'fdd-divide-mwkr':
                candidate_masked = candidate_oprs[np.where(~mask)]
                fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                priority = fdd / wkr
                idx = np.random.choice(np.where(priority == np.min(priority))[0])
                action = candidate_masked[idx]
            else:
                assert print('select "spt" or "fdd-divide-mwkr".')
                action = None
            actions.append(action)

            permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart,
                                 opIDsOnMchs=opIDsOnMchs)

            # update action space or mask
            if action not in last_col:
                candidate_oprs[action // n_mch] += 1
            else:
                mask[action // n_mch] = 1
            # update finished_mark:
            finished_mark[action // n_mch, action % n_mch] = 1
        for i in range(opIDsOnMchs.shape[1] - 1):
            adj[opIDsOnMchs[:, i + 1], opIDsOnMchs[:, i]] = 1

        # forward and backward pass
        earliest_st, latest_st, adj_mat_aug, G = forward_and_backward_pass(adj, dur_mat, plot_G=plot)

        earliest_start = earliest_st.astype(np.float32)
        latest_start = latest_st.astype(np.float32)
        f1 = torch.from_numpy(
            np.pad(np.float32((instance[0].reshape(-1, 1)) / self.high), ((1, 1), (0, 0)), 'constant',
                   constant_values=0))
        f2 = torch.from_numpy(earliest_start.reshape(-1, 1) / 1000)
        f3 = torch.from_numpy(latest_start.reshape(-1, 1) / 1000)
        x = torch.cat([f1, f2, f3], dim=-1)
        edge_idx = torch.nonzero(torch.from_numpy(adj_mat_aug)).t().contiguous()
        init_state = Data(x=x, edge_index=edge_idx, y=np.amax(earliest_start))
        return init_state, G'''

    def show_state(self, G):
        x_axis = np.pad(np.tile(np.arange(1, self.n_mch + 1, 1), self.n_job), (1, 1), 'constant',
                        constant_values=[0, self.n_mch + 1])
        y_axis = np.pad(np.arange(self.n_job, 0, -1).repeat(self.n_mch), (1, 1), 'constant',
                        constant_values=np.median(np.arange(self.n_job, 0, -1)))
        pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
        nx.draw(
            G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'
            # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
        )
        plt.show()

    def _p_list_solver(self, args, plot=False):
        instances, priority_lists, device = args[0], args[1], args[2]

        edge_indices = []
        durations = []
        current_graphs = []
        for i, item in enumerate(zip(instances, priority_lists)):
            instance, priority_list = item[0], item[1]
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            # prepare NIPS adj
            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
            list_for_latest_task_onMachine = [None] * n_machines  # Init list_for_latest_task_onMachine
            adj_mat = np.eye(n_operations, k=-1, dtype=int)  # Create adjacent matrix for the corresponding action list
            adj_mat[np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,
                    0]] = 0  # first column does not have upper stream conj_nei
            # Construct NIPS adjacent matrix
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)

            # prepare augmented adj, augmented dur, and G
            adj_mat = np.pad(adj_mat, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
            adj_mat[[i for i in range(1, n_jobs * n_machines + 2 - 1,
                                      n_machines)], 0] = 1  # connect S with 1st operation of each job
            adj_mat[-1, [i for i in range(n_machines, n_jobs * n_machines + 2 - 1,
                                          n_machines)]] = 1  # connect last operation of each job to T
            adj_mat = np.transpose(adj_mat)  # convert input adj from column pointing to row, to, row pointing to column
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(
                n_jobs * n_machines + 2, axis=1)
            edge_weight = np.multiply(adj_mat, dur_mat)
            G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, n_jobs * n_machines + 2 - 1,
                                                                n_machines)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            if plot:
                self.show_state(G)

            edge_indices.append(
                torch.nonzero(torch.from_numpy(adj_mat)).t().contiguous().to(device) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))
            current_graphs.append(G)

        edge_indices = torch.cat(edge_indices, dim=-1)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        est, lst, make_span = self.eva.forward(edge_index=edge_indices, duration=durations, n_j=self.n_job, n_m=self.n_mch)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(
            np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)
        return (x, torch_geometric.utils.add_self_loops(edge_indices)[0], batch), current_graphs, make_span

    def _rules_solver(self, args, plot=False):
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices = []
        durations = []
        current_graphs = []
        for i, instance in enumerate(instances):
            dur_mat, dur_cp, mch_mat = instance[0], np.copy(instance[0]), instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            first_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, 0]
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,
                             0]  # initialize action space: [n_jobs, 1], the first column
            mask = np.zeros(shape=n_jobs, dtype=bool)  # initialize the mask: [n_jobs, 1]
            conj_nei_up_stream = np.eye(n_operations, k=-1, dtype=np.single)  # initialize adj matrix
            conj_nei_up_stream[first_col] = 0  # first column does not have upper stream conj_nei
            adj_mat = conj_nei_up_stream

            gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            opIDsOnMchs = -n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

            actions = []
            for _ in range(n_operations):

                if rule_type == 'spt':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    dur_candidate = np.take(dur_mat, candidate_masked)
                    idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                    action = candidate_masked[idx]
                elif rule_type == 'fdd-divide-mwkr':
                    candidate_masked = candidate_oprs[np.where(~mask)]
                    fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                    wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                    priority = fdd / wkr
                    idx = np.random.choice(np.where(priority == np.min(priority))[0])
                    action = candidate_masked[idx]
                else:
                    action = None
                actions.append(action)

                permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart,
                                     opIDsOnMchs=opIDsOnMchs)

                # update action space or mask
                if action not in last_col:
                    candidate_oprs[action // n_machines] += 1
                else:
                    mask[action // n_machines] = 1
                # update finished_mark:
                finished_mark[action // n_machines, action % n_machines] = 1

            for _ in range(opIDsOnMchs.shape[1] - 1):
                adj_mat[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

            # prepare augmented adj, augmented dur, and G
            adj_mat = np.pad(adj_mat, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
            adj_mat[[i for i in range(1, n_jobs * n_machines + 2 - 1,
                                      n_machines)], 0] = 1  # connect S with 1st operation of each job
            adj_mat[-1, [i for i in range(n_machines, n_jobs * n_machines + 2 - 1,
                                          n_machines)]] = 1  # connect last operation of each job to T
            adj_mat = np.transpose(adj_mat)  # convert input adj from column pointing to row, to, row pointing to column
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(
                n_jobs * n_machines + 2, axis=1)
            edge_weight = np.multiply(adj_mat, dur_mat)
            G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, n_jobs * n_machines + 2 - 1,
                                                                n_machines)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            if plot:
                self.show_state(G)

            edge_indices.append(
                torch.nonzero(torch.from_numpy(adj_mat)).t().contiguous().to(device) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))
            current_graphs.append(G)

        edge_indices = torch.cat(edge_indices, dim=-1)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        est, lst, make_span = self.eva.forward(edge_index=edge_indices, duration=durations, n_j=self.n_job, n_m=self.n_mch)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(
            np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, torch_geometric.utils.add_self_loops(edge_indices)[0], batch), current_graphs, make_span

    def dag2pyg(self, G, instance):
        n_job, n_mch = instance[0].shape[0], instance[0].shape[1]
        n_oprs = n_job * n_mch

        # start to build PyG data
        adj = nx.to_numpy_matrix(G)
        adj[0, [i for i in range(1, n_oprs + 2 - 1, n_mch)]] = 1
        np.fill_diagonal(adj, 1)
        topological_order = list(nx.topological_sort(G))
        est_ST = np.fromiter(jsp.forward_pass(graph=G, topological_order=topological_order).values(), dtype=np.float32)
        lst_ST = np.fromiter(jsp.backward_pass(graph=G, topological_order=topological_order, makespan=est_ST[-1]).values(), dtype=np.float32)
        f1 = torch.from_numpy(np.pad(np.float32((instance[0].reshape(-1, 1)) / self.high), ((1, 1), (0, 0)), 'constant', constant_values=0))
        f2 = torch.from_numpy(est_ST.reshape(-1, 1) / self.fea_norm_const)
        f3 = torch.from_numpy(lst_ST.reshape(-1, 1) / self.fea_norm_const)
        x = torch.cat([f1, f2, f3], dim=-1)
        edge_idx = torch.nonzero(torch.from_numpy(adj)).t().contiguous()
        return Data(x=x, edge_index=edge_idx, y=np.amax(est_ST))

    def _transit_single(self, plot, args):
        """
        action: [2,]
        """
        action, sol, instance = args[0], args[1], args[2]

        if action == [0, 0]:  # if dummy action then do not transit
            return dag2pyg(G=sol, instance=instance, high=self.high, min_max=self.min_max, normalizer=self.normalizer)
        else:
            S = [s for s in sol.predecessors(action[0]) if
                 int((s - 1) // self.n_mch) != int((action[0] - 1) // self.n_mch) and s != 0]
            T = [t for t in sol.successors(action[1]) if
                 int((t - 1) // self.n_mch) != int((action[1] - 1) // self.n_mch) and t != self.n_oprs + 1]
            s = S[0] if len(S) != 0 else None
            t = T[0] if len(T) != 0 else None

            if s is not None:  # connect s with action[1]
                sol.remove_edge(s, action[0])
                sol.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
            else:
                pass

            if t is not None:  # connect action[0] with t
                sol.remove_edge(action[1], t)
                sol.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
            else:
                pass

            # reverse edge connecting selected pair
            sol.remove_edge(action[0], action[1])
            sol.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

            new_state = dag2pyg(G=sol, instance=instance, high=self.high, min_max=self.min_max,
                                normalizer=self.normalizer)

            if plot:
                plot_sol(sol, n_job=self.n_job, n_mch=self.n_mch)

            return new_state

    def reset(self, instances, init_type, device, plot=False):
        self.instances = instances
        if init_type == 'plist':
            random_plist = np.repeat(np.arange(self.n_job).repeat(self.n_mch).reshape(1, -1), repeats=self.instances.shape[0], axis=0)  # fixed priority list: [0, 0, 0, ..., n-1, n-1, n-1]
            (x, edge_indices, batch), current_graphs, make_span = self._p_list_solver(args=[self.instances, random_plist, device], plot=plot)
        elif init_type == 'spt':
            (x, edge_indices, batch), current_graphs, make_span = self._rules_solver(args=[self.instances, device, 'spt'], plot=plot)
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices, batch), current_graphs, make_span = self._rules_solver(args=[self.instances, device, 'fdd-divide-mwkr'], plot=plot)
        else:
            assert False, 'Initial solution type = "p_list", "spt", "fdd-divide-mwkr".'

        # print(x)
        # print(edge_indices)
        # print(batch)

        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(instances.shape[0])]
        feasible_actions, flag = self.feasible_actions()
        if self.itr == self.max_transition:
            done = True
        else:
            done = False

        return (x, edge_indices, batch), feasible_actions, done

    def feasible_actions(self):
        actions = []
        feasible_actions_flag = []  # 0 for no feasible operation pairs
        for current_graph, instance, tabu_list in zip(self.current_graphs, self.instances, self.tabu_lists):
            action = self._gen_moves(solution=current_graph, mch_mat=instance[1], tabu_list=tabu_list)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(1)
            else:  # if no feasible actions available append dummy actions [0, 0], this is the sign for local search terminate
                actions.append([[0, 0]])
                feasible_actions_flag.append(0)
        return actions, np.array(feasible_actions_flag)



def main():
    from torch_geometric.data.batch import Batch
    from ortools_baseline import MinimalJobshopSat
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    np.random.seed(3)  # 123456324

    j = 3
    m = 3
    h = 99
    l = 1
    transit = 1
    batch_size = 2

    # insts = np.load('../test_data/tai{}x{}.npy'.format(j, m))[:1]
    insts = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, init='p_list', rule='fdd-divide-mwkr', transition=transit)
    env.reset(instances=insts, init_type='spt', device=device)


if __name__ == '__main__':
    import time

    t1 = time.time()
    main()
    print('main() function running time:', time.time() - t1)
