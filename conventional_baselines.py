import numpy as np
import torch
import time
from env.env_batch import JsspN5
import copy
import random

show = False

# problem config
p_j = 15
p_m = 15
l = 1
h = 99
testing_type = 'tai'  # 'tai', 'syn'
n_generated_instances = 100
init = 'spt'  # 'fdd-divide-mwkr', 'spt', ...
reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'

# MDP config
transit = 1000
result_type = 'incumbent'  # 'current', 'incumbent'

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def best_improvement_move(support_env, feasible_actions, current_graph, current_tabu_list, current_obj, incumbent_obj,
                          instance, device):
    # only support single instance, so env.inst.shape = [b=1, 2, j, m]

    n_feasible_actions = len(feasible_actions)

    duplicated_instances = np.tile(instance, reps=[n_feasible_actions, 1, 1, 1])
    duplicated_current_obj = current_obj.repeat(n_feasible_actions, 1)
    duplicated_incumbent_obj = incumbent_obj.repeat(n_feasible_actions, 1)
    duplicated_current_graphs = [copy.deepcopy(current_graph) for _ in range(n_feasible_actions)]
    duplicated_tabu_lists = [copy.copy(current_tabu_list) for _ in range(n_feasible_actions)]

    support_env.instances = duplicated_instances
    support_env.current_graphs = duplicated_current_graphs
    support_env.current_objs = duplicated_current_obj
    support_env.tabu_lists = duplicated_tabu_lists
    support_env.incumbent_objs = duplicated_incumbent_obj

    support_env.step(feasible_actions, device)

    if support_env.current_objs.min().cpu().item() < current_obj.cpu().item():
        best_move = [feasible_actions[torch.argmin(support_env.current_objs, dim=0, keepdim=True).cpu().item()]]
    else:
        best_move = [random.choice(feasible_actions)]
        # best_move = [[0, 0]]

    return best_move


def tabu_move(support_env, feasible_actions, current_graph, current_tabu_list, current_obj, incumbent_obj, instance,
              device):
    # only support single instance, so env.inst.shape = [b=1, 2, j, m]

    n_feasible_actions = len(feasible_actions)

    duplicated_instances = np.tile(instance, reps=[n_feasible_actions, 1, 1, 1])
    duplicated_current_obj = current_obj.repeat(n_feasible_actions, 1)
    duplicated_incumbent_obj = incumbent_obj.repeat(n_feasible_actions, 1)
    duplicated_current_graphs = [copy.deepcopy(current_graph) for _ in range(n_feasible_actions)]
    duplicated_tabu_lists = [copy.copy(current_tabu_list) for _ in range(n_feasible_actions)]

    support_env.instances = duplicated_instances
    support_env.current_graphs = duplicated_current_graphs
    support_env.current_objs = duplicated_current_obj
    support_env.tabu_lists = duplicated_tabu_lists
    support_env.incumbent_objs = duplicated_incumbent_obj

    support_env.step(feasible_actions, device)

    tabu_mv = [feasible_actions[torch.argmin(support_env.current_objs, dim=0, keepdim=True).cpu().item()]]

    return tabu_mv


def main():
    env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type=reward_type)
    inst = np.load('./test_data/{}{}x{}.npy'.format(testing_type, p_j, p_m))

    compare_against = np.load('./test_data/{}{}x{}_result.npy'.format(testing_type, p_j, p_m))
    print('compare against:', compare_against)

    support_env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin')  # reward_type doesn't matter

    # rollout best_improvement_move
    print('Starting rollout best_improvement_move policy...')
    t1_best_improvement = time.time()
    best_improvement_result = []
    for ins in inst[:]:
        ins = np.array([ins])
        _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)

        while env.itr < transit:
            best_move = best_improvement_move(support_env=support_env,
                                              feasible_actions=feasible_actions[0],
                                              current_graph=env.current_graphs[0],
                                              current_tabu_list=env.tabu_lists[0],
                                              current_obj=env.current_objs[0],
                                              incumbent_obj=env.incumbent_objs[0],
                                              instance=env.instances[0],
                                              device=dev)
            _, _, feasible_actions, _ = env.step(best_move, dev)
        print(env.incumbent_objs.cpu().item())
        best_improvement_result.append(env.incumbent_objs.cpu().item())
    t2_best_improvement = time.time()
    best_improvement_result = np.array(best_improvement_result)
    print(best_improvement_result)
    print('Best_improvement_move results takes: {:.4f}s per instance.\n'.format((t2_best_improvement - t1_best_improvement)/inst.shape[0]), best_improvement_result)
    print('Gap:', ((best_improvement_result - compare_against) / compare_against).mean())

    # rollout tabu_move
    print('Starting rollout tabu_move policy...')
    t1_tabu = time.time()
    tabu_result = []
    for ins in inst[:]:
        ins = np.array([ins])
        _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)

        while env.itr < transit:
            best_move = tabu_move(support_env=support_env,
                                  feasible_actions=feasible_actions[0],
                                  current_graph=env.current_graphs[0],
                                  current_tabu_list=env.tabu_lists[0],
                                  current_obj=env.current_objs[0],
                                  incumbent_obj=env.incumbent_objs[0],
                                  instance=env.instances[0],
                                  device=dev)
            _, _, feasible_actions, _ = env.step(best_move, dev)
        print(env.incumbent_objs.cpu().item())
        tabu_result.append(env.incumbent_objs.cpu().item())
    t2_tabu = time.time()
    tabu_result = np.array(tabu_result)
    print(tabu_result)
    print('Tabu_move results takes: {:.4f}s per instance.\n'.format(t2_tabu - t1_tabu/inst.shape[0]), tabu_result)
    print('Gap:', ((tabu_result - compare_against) / compare_against).mean())


if __name__ == '__main__':
    main()
