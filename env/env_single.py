import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy
import torch
import numpy as np
import networkx as nx
from env.generateJSP import uni_instance_gen
import env.jsp_problem as jsp
from sklearn.preprocessing import MinMaxScaler
from env.utils import plot_sol, dag2pyg
from torch_geometric.data import Data
from env.permissible_LS import permissibleLeftShift
from parameters import args as parameters
from env.jsp_problem import forward_and_backward_pass


class JsspN5:
    def __init__(self,
                 n_job,
                 n_mch,
                 low,
                 high,
                 init='p_list',
                 rule='spt',
                 min_max=False,
                 transition=100):

        self.n_job = n_job
        self.n_mch = n_mch
        self.n_oprs = self.n_job * self.n_mch
        self.low = low
        self.high = high
        self.init = init
        self.rule = rule
        self.min_max = min_max
        self.itr = 0
        self.max_transition = transition
        self.instance = None
        self.current_graph = None
        self.current_objs = None
        self.normalizer = MinMaxScaler()
        self.tabu_size = 1
        self.tabu_list = []
        self.incumbent_obj = None
        self.incumbent_idle = None

    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        """
        solution: networkx DAG conjunctive graph
        mch_mat: the same mch from our NeurIPS 2020 paper of solution
        """
        critical_path = nx.dag_longest_path(solution)[1:-1]
        critical_blocks_opr = np.array(critical_path)
        critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 1
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

    def _p_list_solver_single_instance(self, plot, args):
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
            np.pad(np.float32((instance[0].reshape(-1, 1)) / self.high), ((1, 1), (0, 0)), 'constant', constant_values=0))
        if self.min_max:
            self.normalizer.fit(earliest_start.reshape(-1, 1))
            f2 = torch.from_numpy(self.normalizer.transform(earliest_start.reshape(-1, 1)))
            self.normalizer.fit(latest_start.reshape(-1, 1))
            f3 = torch.from_numpy(self.normalizer.transform(latest_start.reshape(-1, 1)))
        else:
            f2 = torch.from_numpy(earliest_start.reshape(-1, 1)/1000)
            f3 = torch.from_numpy(latest_start.reshape(-1, 1)/1000)
        x = torch.cat([f1, f2, f3], dim=-1)
        edge_idx = torch.nonzero(torch.from_numpy(adj_aug)).t().contiguous()
        init_state = Data(x=x, edge_index=edge_idx, y=np.amax(earliest_start))
        return init_state, G

    def rules_solver(self, instance, plot=False):
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

        gant_chart = -parameters.h * np.ones_like(dur_mat.transpose(), dtype=np.int32)
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

            permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart, opIDsOnMchs=opIDsOnMchs)

            # update action space or mask
            if action not in last_col:
                candidate_oprs[action // n_mch] += 1
            else:
                mask[action // n_mch] = 1
            # update finished_mark:
            finished_mark[action // n_mch, action % n_mch] = 1
        for i in range(opIDsOnMchs.shape[1] - 1):
            adj[opIDsOnMchs[:, i+1], opIDsOnMchs[:, i]] = 1

        # forward and backward pass
        earliest_st, latest_st, adj_mat_aug, G = forward_and_backward_pass(adj, dur_mat, plot_G=plot)

        earliest_start = earliest_st.astype(np.float32)
        latest_start = latest_st.astype(np.float32)
        f1 = torch.from_numpy(
            np.pad(np.float32((instance[0].reshape(-1, 1)) / self.high), ((1, 1), (0, 0)), 'constant',
                   constant_values=0))
        if self.min_max:
            self.normalizer.fit(earliest_start.reshape(-1, 1))
            f2 = torch.from_numpy(self.normalizer.transform(earliest_start.reshape(-1, 1)))
            self.normalizer.fit(latest_start.reshape(-1, 1))
            f3 = torch.from_numpy(self.normalizer.transform(latest_start.reshape(-1, 1)))
        else:
            f2 = torch.from_numpy(earliest_start.reshape(-1, 1) / 1000)
            f3 = torch.from_numpy(latest_start.reshape(-1, 1) / 1000)
        x = torch.cat([f1, f2, f3], dim=-1)
        edge_idx = torch.nonzero(torch.from_numpy(adj_mat_aug)).t().contiguous()
        init_state = Data(x=x, edge_index=edge_idx, y=np.amax(earliest_start))
        return init_state, G

    def _transit_single(self, plot, args):
        """
        action: [2,]
        """
        action, sol, instance = args[0], args[1], args[2]

        if action == [0, 0]:  # if dummy action then do not transit
            return dag2pyg(G=sol, instance=instance, high=self.high, min_max=self.min_max, normalizer=self.normalizer)
        else:
            S = [s for s in sol.predecessors(action[0]) if int((s-1)//self.n_mch) != int((action[0]-1)//self.n_mch) and s != 0]
            T = [t for t in sol.successors(action[1]) if int((t-1)//self.n_mch) != int((action[1]-1)//self.n_mch) and t != self.n_oprs+1]
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

    def _instances_gen(self):
        return numpy.stack(uni_instance_gen(self.n_job, self.n_mch, self.low, self.high))

    def _init(self, plot=False):
        if self.init == 'p_list':
            # p_list = np.random.permutation(np.arange(self.n_job).repeat(self.n_mch))
            p_list = np.arange(self.n_job).repeat(self.n_mch)  # fixed priority list: [0, 0, 0, ..., n-1, n-1, n-1]
            data, G = self._p_list_solver_single_instance(plot, args=[self.instance, p_list])
            return data, G
        elif self.init == 'rule':
            data, G = self.rules_solver(self.instance)
            return data, G
        else:
            print('env.init = "p_list" or "rule". ')

    def reset(self, instance=None, fix_instance=False, plot=False):
        if fix_instance:
            self.instance = instance
        else:
            self.instance = self._instances_gen()
        init_state, init_graph = self._init(plot)
        self.current_graph = init_graph
        self.current_objs = init_state.y
        self.incumbent_obj = init_state.y
        self.itr = 0
        self.tabu_list = []
        feasible_actions = self.feasible_action()
        if self.itr == self.max_transition or len(feasible_actions) == 0:
            done = True
        else:
            done = False
        return init_state, feasible_actions, done

    def feasible_action(self):
        action = self._gen_moves(solution=self.current_graph, mch_mat=self.instance[1], tabu_list=self.tabu_list)
        return action

    def step_single(self, action, plot=False):
        new_state = self._transit_single(plot, args=[action, self.current_graph, self.instance])

        # makespan reward
        diff1 = torch.tensor(self.current_objs) - torch.tensor(new_state.y)
        reward = diff1
        self.incumbent_obj = np.where(np.array(new_state.y) < self.incumbent_obj, new_state.y, self.incumbent_obj)
        self.current_objs = new_state.y

        # sequential version of update tabu list, different tabu list can have different length
        if self.tabu_size != 0:
            action_reversed = action[::-1]
            if action_reversed == [0, 0]:  # if dummy action, don't update tabu list
                pass
            else:
                if len(self.tabu_list) == self.tabu_size:
                    self.tabu_list.pop(0)
                    self.tabu_list.append(action_reversed)
                else:
                    self.tabu_list.append(action_reversed)

        self.itr = self.itr + 1

        feasible_actions = self.feasible_action()
        if self.itr == self.max_transition or len(feasible_actions) == 0:
            done = True
        else:
            done = False

        return new_state, reward, feasible_actions, done


def main():
    from torch_geometric.data.batch import Batch
    from ortools_baseline import MinimalJobshopSat
    import torch_geometric.utils
    from model.actor import Actor
    import random

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(3)  # 123456324

    j = 10
    m = 10
    h = 99
    l = 1
    transit = 2000
    batch_size = 1

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h,
                 init='rule', rule='fdd-divide-mwkr', transition=transit)
    actor = Actor(in_dim=3, hidden_dim=64).to(device).eval()

    # inst = np.expand_dims(np.load('./test_inst.npy')[6], axis=0)
    # inst = np.load('../test_data/tai{}x{}.npy'.format(j, m))[:batch_size]
    inst = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    # saved_acts = np.load('./saved_acts.npy')

    # print([param for param in actor.parameters()])
    # print(inst)

    initial_gap = []
    simulate_result = []
    for i, data in enumerate(inst):
        state, feasible_action, done = env.reset(instance=data, fix_instance=True)
        initial_gap.append(env.current_objs)
        print('Initial sol:', env.current_objs)
        returns = []
        t = 0
        with torch.no_grad():
            while not done:
                if state.edge_index.shape[1] != (j-1)*m + (m-1)*j + (j*m+2) + j + j:
                    print('not equal {} at:'.format((j-1)*m + (m-1)*j + (j*m+2) + j + j), env.itr)
                    np.save('./mal_func_instance.npy', env.instance)
                # action = [random.choice(feasible_action)]
                # action = np.expand_dims(saved_acts[env.itr], axis=0).tolist()
                batch_data = Batch.from_data_list([state]).to(device)
                action, _ = actor(batch_data, [feasible_action])

                # print(Batch.from_data_list([state]).to(device).x)
                # print(torch_geometric.utils.sort_edge_index(Batch.from_data_list([state]).to(device).edge_index)[0])
                print(action[0])

                state_prime, reward, new_feasible_actions, done = env.step_single(action=action[0])
                # print('make span reward:', reward)
                if torch.equal(state.x.cpu(), state_prime.x) and torch.equal(state.edge_index.cpu(), state_prime.edge_index):
                    print('In absorbing state at', env.itr - 1)
                returns.append(reward)
                state = state_prime
                feasible_action = new_feasible_actions

                t += 1
                # print()
        simulate_result.append(env.incumbent_obj)
        print('Incumbent sol:', env.incumbent_obj)
        print('Instance-' + str(i + 1) + ' ends after ' + str(env.itr) + ' transitions')
        print()
    simulate_result = np.array(simulate_result)
    initial_gap = np.array(initial_gap)

    '''# ortools solver
    results_ortools = []
    for i, data in enumerate(inst):
        times_rearrange = np.expand_dims(data[0], axis=-1)
        machines_rearrange = np.expand_dims(data[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        result = MinimalJobshopSat(data.tolist())
        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
        results_ortools.append(result[1])
    results_ortools = np.array(results_ortools)

    print('Initial Gap:', ((initial_gap - results_ortools) / results_ortools).mean())
    print('Simulation Gap:', ((simulate_result - results_ortools) / results_ortools).mean())'''


if __name__ == '__main__':
    import time

    t1 = time.time()
    main()
    # print('\nmain() function running time:', time.time() - t1)


