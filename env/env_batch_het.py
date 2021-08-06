import os
import sys

from torch_geometric.utils import add_self_loops

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import networkx as nx
from env.generateJSP import uni_instance_gen
from env.permissible_LS import permissibleLeftShift
from env.message_passing_evl import Evaluator
import matplotlib.pyplot as plt
import time
import random
from model.actor import Actor


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None

    def wrapper(self, x, edge_index_pc, edge_index_mc, batch):
        self.x = x
        self.edge_index_pc = edge_index_pc
        self.edge_index_mc = edge_index_mc
        self.batch = batch

    def clean(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None


class JsspN5:
    def __init__(self, n_job, n_mch, low, high, reward_type, fea_norm_const=1000):

        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        self.itr = 0
        self.instances = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.reward_type = reward_type
        self.fea_norm_const = fea_norm_const
        self.eva = Evaluator()
        self.adj_mat_pc = self._adj_mat_pc()


    def _adj_mat_pc(self):
        adj_mat_pc = np.eye(self.n_oprs, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
        adj_mat_pc[np.arange(start=0, stop=self.n_oprs, step=1).reshape(self.n_job, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
        adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
        adj_mat_pc[[i for i in range(1, self.n_job * self.n_mch + 2 - 1, self.n_mch)], 0] = 1  # connect S with 1st operation of each job
        adj_mat_pc[-1, [i for i in range(self.n_mch, self.n_job * self.n_mch + 2 - 1, self.n_mch)]] = 1  # connect last operation of each job to T
        adj_mat_pc = np.transpose(adj_mat_pc)  # convert input adj from column pointing to row, to, row pointing to column
        return adj_mat_pc


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

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        for i, (instance, priority_list) in enumerate(zip(instances, priority_lists)):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            # prepare NIPS adj
            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
            list_for_latest_task_onMachine = [None] * n_machines  # Init list_for_latest_task_onMachine

            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique
            # Construct NIPS adjacent matrix
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat_mc[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(
                n_jobs * n_machines + 2, axis=1)
            edge_weight = np.multiply((self.adj_mat_pc + adj_mat_mc), dur_mat)
            G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, n_jobs * n_machines + 2 - 1,
                                                                n_machines)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.tensor(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.tensor(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))
            current_graphs.append(G)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)

        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(
            np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, make_span


    def _rules_solver(self, args, plot=False):
        instances, device, rule_type = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        for i, instance in enumerate(instances):
            dur_mat, dur_cp, mch_mat = instance[0], np.copy(instance[0]), instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,0]  # initialize action space: [n_jobs, 1], the first column
            mask = np.zeros(shape=n_jobs, dtype=bool)  # initialize the mask: [n_jobs, 1]
            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique

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
                adj_mat_mc[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

            # prepare augmented adj, augmented dur, and G
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant',
                                constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(n_jobs * n_machines + 2, axis=1)
            edge_weight = np.multiply((self.adj_mat_pc + adj_mat_mc), dur_mat)
            G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, n_jobs * n_machines + 2 - 1,
                                                                n_machines)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.tensor(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.tensor(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))
            current_graphs.append(G)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, make_span

    def dag2pyg(self, instances, nx_graphs, device):
        n_jobs, n_machines = instances[0][0].shape
        n_operations = n_jobs * n_machines

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        for i, (instance, G) in enumerate(zip(instances, nx_graphs)):
            durations.append(np.pad(instance[0].reshape(-1), (1, 1), 'constant', constant_values=0))
            adj_all = nx.adjacency_matrix(G, weight=None).todense()
            adj_mat_mc = adj_all - self.adj_mat_pc
            edge_indices_pc.append((torch.nonzero(torch.tensor(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.tensor(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.from_numpy(np.concatenate(durations)).reshape(-1, 1).to(device)
        est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=n_jobs, n_m=n_machines)
        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=n_jobs * n_machines + 2)).to(device)

        return x, edge_indices_pc, edge_indices_mc, batch, make_span

    def change_nxgraph_topology(self, actions, plot=False):
        n_jobs, n_machines = self.instances[0][0].shape
        n_operations = n_jobs * n_machines

        for i, (action, G, instance) in enumerate(zip(actions, self.current_graphs, self.instances)):
            if action == [0, 0]:  # if dummy action then do not transit
                pass
            else:  # change nx graph topology
                S = [s for s in G.predecessors(action[0]) if
                     int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
                T = [t for t in G.successors(action[1]) if
                     int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
                s = S[0] if len(S) != 0 else None
                t = T[0] if len(T) != 0 else None

                if s is not None:  # connect s with action[1]
                    G.remove_edge(s, action[0])
                    G.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                else:
                    pass

                if t is not None:  # connect action[0] with t
                    G.remove_edge(action[1], t)
                    G.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                else:
                    pass

                # reverse edge connecting selected pair
                G.remove_edge(action[0], action[1])
                G.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))
            if plot:
                self.show_state(G)

    def step(self, actions, device, plot=False):
        self.change_nxgraph_topology(actions, plot=plot)  # change graph topology
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.current_graphs, device)  # generate new state data
        if self.reward_type == 'consecutive':
            reward = self.current_objs - makespan
        elif self.reward_type == 'yaoxin':
            reward = torch.where(self.incumbent_objs - makespan > 0, self.incumbent_objs - makespan, torch.tensor(0, dtype=torch.float32, device=device))
        else:
            raise ValueError('reward type must be "yaoxin" or "consecutive".')

        self.incumbent_objs = torch.where(makespan - self.incumbent_objs < 0, makespan, self.incumbent_objs)
        self.current_objs = makespan

        # update tabu list
        if self.tabu_size != 0:
            action_reversed = [a[::-1] for a in actions]
            for i, action in enumerate(action_reversed):
                if action == [0, 0]:  # if dummy action, don't update tabu list
                    pass
                else:
                    if len(self.tabu_lists[i]) == self.tabu_size:
                        self.tabu_lists[i].pop(0)
                        self.tabu_lists[i].append(action)
                    else:
                        self.tabu_lists[i].append(action)

        self.itr = self.itr + 1


        feasible_actions, flag = self.feasible_actions(device)  # new feasible actions w.r.t updated tabu list

        return (x, edge_indices_pc, edge_indices_mc, batch), reward, feasible_actions, ~flag

    def reset(self, instances, init_type, device, plot=False):
        self.instances = instances
        if init_type == 'plist':
            random_plist = np.repeat(np.arange(self.n_job).repeat(self.n_mch).reshape(1, -1), repeats=self.instances.shape[0], axis=0)  # fixed priority list: [0, 0, 0, ..., n-1, n-1, n-1]
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, make_span = self._p_list_solver(args=[self.instances, random_plist, device], plot=plot)
        elif init_type == 'spt':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, make_span = self._rules_solver(args=[self.instances, device, 'spt'], plot=plot)
        elif init_type == 'fdd-divide-mwkr':
            (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, make_span = self._rules_solver(args=[self.instances, device, 'fdd-divide-mwkr'], plot=plot)
        else:
            assert False, 'Initial solution type = "p_list", "spt", "fdd-divide-mwkr".'

        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(instances.shape[0])]
        feasible_actions, flag = self.feasible_actions(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), feasible_actions, ~flag

    def feasible_actions(self, device):
        actions = []
        feasible_actions_flag = []  # False for no feasible operation pairs
        for i, (current_graph, instance, tabu_list) in enumerate(zip(self.current_graphs, self.instances, self.tabu_lists)):
            action = self._gen_moves(solution=current_graph, mch_mat=instance[1], tabu_list=tabu_list)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:  # if no feasible actions available append dummy actions [0, 0]
                actions.append([[0, 0]])
                feasible_actions_flag.append(False)
        return actions, torch.tensor(feasible_actions_flag, device=device).unsqueeze(1)


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    j = 100
    m = 20
    h = 99
    l = 1
    transit = 128
    batch_size = 10
    n_batch = 1
    init = 'fdd-divide-mwkr'
    reward_type = 'yaoxin'

    # insts = np.load('../test_data/tai{}x{}.npy'.format(j, m))[:batch_size]
    insts = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    # np.save('test_inst.npy', insts)
    # print(insts)
    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, reward_type=reward_type)
    actor = Actor(in_dim=3, hidden_dim=64, embedding_type='gin').to(device)
    # print([param for param in actor.parameters()])

    for b_i in range(n_batch):

        t3 = time.time()
        states, feasible_actions, done = env.reset(instances=insts, init_type=init, device=device)
        batch_wrapper = BatchGraph()
        # print(env.incumbent_objs)

        # print(env.adj_mat_pc)

        saved_acts = []
        returns = []
        n_nodes_per_graph = j * m + 2
        n_edges_per_graph = j*(m-1) + m*(j-1) + j*m+2 + j*2
        with torch.no_grad():
            while env.itr < transit:
                # print(states[2])
                # print(states[1])
                # print(*states)
                batch_wrapper.wrapper(*states)
                # actions, _ = actor(batch_wrapper, feasible_actions)
                actions = [random.choice(feasible_actions[i]) for i in range(len(feasible_actions))]

                states, reward, feasible_actions, done = env.step(actions, device)

                returns.append(reward)

                # print(reward)
                # print(env.itr)
                # print()

            # np.save('saved_acts.npy', np.array(saved_acts))

        t4 = time.time()

        print(t4 - t3)
        print(env.incumbent_objs)

        # print(torch.count_nonzero(torch.cat(returns, dim=-1), dim=-1))


if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(3)  # 123456324

    t1 = time.time()
    main()
    # print('main() function running time:', time.time() - t1)
