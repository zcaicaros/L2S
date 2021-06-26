import os
import random
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
from model.actor import Actor


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
    def _get_pairs(cb, cb_op, tabu_list=None):  # first 2 operations of first block and last 2 operations of last block is also included
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
        dur_mat, mch_mat = instance[0], instance[1]
        n_job, n_mch = dur_mat.shape[0], dur_mat.shape[1]
        n_opr = n_job * n_mch
        last_col = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, -1]
        first_col = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, 0]
        candidate_oprs = np.arange(start=0, stop=n_opr, step=1).reshape(n_job, -1)[:, 0]
        mask = np.zeros(shape=n_job, dtype=bool)
        # initialize adj matrix
        conj_nei_up_stream = np.eye(n_opr, k=-1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[first_col] = 0
        self_as_nei = np.eye(n_opr, dtype=np.single)
        adj = self_as_nei + conj_nei_up_stream
        p_list = []
        if self.rule == 'spt':
            for _ in range(n_opr):
                candidate_masked = candidate_oprs[np.where(~mask)]
                dur_candidate = np.take(dur_mat, candidate_masked)
                idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                action = candidate_masked[idx]
                if action not in last_col:
                    candidate_oprs[action // n_mch] += 1
                else:
                    mask[action // n_mch] = 1
                job_id = action // n_mch
                p_list.append(job_id)
        data, G = self._p_list_solver_single_instance(plot, args=[self.instance, p_list])
        return data, G

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
            p_list = np.random.permutation(np.arange(self.n_job).repeat(self.n_mch))
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
        self.incumbent_idle = compute_idle(state=init_state,
                                           machine_mat=self.instance[1],
                                           dur_mat=self.instance[0],
                                           n_machine=self.n_mch,
                                           n_job=self.n_job)
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
        # diff1 = torch.tensor(self.current_objs) - torch.tensor(new_state.y)
        # reward1 = diff1
        diff1 = torch.tensor(self.incumbent_obj) - torch.tensor(new_state.y)
        reward1 = torch.where(diff1 > 0, diff1/10, torch.tensor(0, dtype=torch.float32))
        self.incumbent_obj = np.where(np.array(new_state.y) < self.incumbent_obj, new_state.y, self.incumbent_obj)
        self.current_objs = new_state.y
        # idle time reward
        new_idle = compute_idle(state=new_state, machine_mat=self.instance[1], dur_mat=self.instance[0], n_machine=self.n_mch, n_job=self.n_job)
        diff2 = self.incumbent_idle - new_idle
        reward2 = torch.where(diff2 > 0, diff2, torch.tensor(0, dtype=torch.float32))
        if self.incumbent_idle > new_idle:
            self.incumbent_idle = new_idle
        # total reward
        reward = reward1 + reward2
        # reward = reward1

        # print(reward1)
        # print(reward2)

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
    import random
    from torch_geometric.data.batch import Batch
    actor = Actor(in_dim=3, hidden_dim=64).to(device)
    state, feasible_action, done = env.reset(plot=plt)
    env.rules_solver(env.instance)
    # np.save('./instances{}x{}.npy'.format(str(n_j), str(n_m)), env.instance)
    returns = []
    t = 0
    with torch.no_grad():
        while not done:
            action, _ = actor(Batch.from_data_list([state]).to(device), [feasible_action])
            # action = random.choice(feasible_action)

            state_prime, reward, new_feasible_actions, done = env.step_single(action=action[0], plot=plt)
            # print('make span reward:', reward)
            if torch.equal(state.x.cpu(), state_prime.x) and torch.equal(state.edge_index.cpu(), state_prime.edge_index):
                print('In absorbing state at', env.itr - 1)

            returns.append(reward)
            state = state_prime
            feasible_action = new_feasible_actions
            t += 1
            # print()
    print(torch.stack(returns).nonzero().shape)


def compute_idle(state,  machine_mat, dur_mat, n_machine, n_job):
    start_time_reshape = state.x[1:-1, 1].reshape(n_job, n_machine) * 1000
    end_time_reshape = start_time_reshape + torch.tensor(dur_mat, dtype=torch.float)
    gant_start_time, _ = torch.sort(torch.stack([start_time_reshape[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    gant_end_time, _ = torch.sort(torch.stack([end_time_reshape[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    mean_idle = (gant_start_time[:, 1:] - gant_end_time[:, :-1]).mean()
    return mean_idle


def new_reward(state, state_prime, machine_mat, dur_mat, n_machine, n_job, incumb_idle):
    start_time_reshape = state.x[1:-1, 1].reshape(n_job, n_machine) * 1000
    end_time_reshape = start_time_reshape + torch.tensor(dur_mat, dtype=torch.float)
    gant_start_time, _ = torch.sort(torch.stack([start_time_reshape[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    gant_end_time, _ = torch.sort(torch.stack([end_time_reshape[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    idle = (gant_start_time[:, 1:] - gant_end_time[:, :-1]).mean()

    '''start_time_reshape_prime = state_prime.x[1:-1, 1].reshape(n_job, n_machine) * 1000
    end_time_reshape_prime = start_time_reshape_prime + torch.tensor(dur_mat, dtype=torch.float)
    gant_start_time_prime, _ = torch.sort(torch.stack([start_time_reshape_prime[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    gant_end_time_prime, _ = torch.sort(torch.stack([end_time_reshape_prime[np.where(machine_mat == i + 1)] for i in range(n_machine)]), dim=-1)
    idle_prime = (gant_start_time_prime[:, 1:] - gant_end_time_prime[:, :-1]).mean()'''

    if idle < incumb_idle:
        # print('idle reward:', incumb_idle - idle)
        return idle
    else:
        # print('idle reward:', torch.tensor(0))
        return incumb_idle


if __name__ == '__main__':
    ###### WHEN COMPUTE USING PARALLEL, env.current_graphs WILL NOT TRANSIT, BUG!!!
    import time

    n_j = 10
    n_m = 10
    l = 1
    h = 99
    transit = 100
    par = False
    plt = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    np.random.seed(1)

    env = JsspN5(n_job=n_j, n_mch=n_m, low=l, high=h, init='rule', rule='spt', transition=transit)

    '''import cProfile
    cProfile.run('main()', filename='./restats_{}x{}_{}'.format(str(n_j), str(n_m), str(env.max_transition)))'''

    t1 = time.time()
    main()
    print(time.time() - t1)
