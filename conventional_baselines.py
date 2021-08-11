import numpy as np
import torch
import time
from env.env_batch import JsspN5
import copy
import random
from ortools_baseline import MinimalJobshopSat


def best_improvement_move(support_env,
                          feasible_actions,
                          current_graph,
                          current_sub_graphs_mc,
                          current_tabu_list,
                          current_obj,
                          incumbent_obj,
                          instance,
                          device):
    # only support single instance, so env.inst.shape = [b=1, 2, j, m]

    n_feasible_actions = len(feasible_actions)

    duplicated_instances = np.tile(instance, reps=[n_feasible_actions, 1, 1, 1])
    duplicated_current_obj = current_obj.repeat(n_feasible_actions, 1)
    duplicated_incumbent_obj = incumbent_obj.repeat(n_feasible_actions, 1)
    duplicated_current_sub_graphs_mc = [copy.deepcopy(current_sub_graphs_mc) for _ in range(n_feasible_actions)]
    duplicated_current_graphs = [copy.deepcopy(current_graph) for _ in range(n_feasible_actions)]
    duplicated_tabu_lists = [copy.copy(current_tabu_list) for _ in range(n_feasible_actions)]

    support_env.instances = duplicated_instances
    support_env.current_graphs = duplicated_current_graphs
    support_env.sub_graphs_mc = duplicated_current_sub_graphs_mc
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


def tabu_move(support_env,
              feasible_actions,
              current_graph,
              current_sub_graphs_mc,
              current_tabu_list,
              current_obj,
              incumbent_obj,
              instance,
              device):
    # only support single instance, so env.inst.shape = [b=1, 2, j, m]

    n_feasible_actions = len(feasible_actions)

    duplicated_instances = np.tile(instance, reps=[n_feasible_actions, 1, 1, 1])
    duplicated_current_obj = current_obj.repeat(n_feasible_actions, 1)
    duplicated_incumbent_obj = incumbent_obj.repeat(n_feasible_actions, 1)
    duplicated_current_sub_graphs_mc = [copy.deepcopy(current_sub_graphs_mc) for _ in range(n_feasible_actions)]
    duplicated_current_graphs = [copy.deepcopy(current_graph) for _ in range(n_feasible_actions)]
    duplicated_tabu_lists = [copy.copy(current_tabu_list) for _ in range(n_feasible_actions)]

    support_env.instances = duplicated_instances
    support_env.current_graphs = duplicated_current_graphs
    support_env.sub_graphs_mc = duplicated_current_sub_graphs_mc
    support_env.current_objs = duplicated_current_obj
    support_env.tabu_lists = duplicated_tabu_lists
    support_env.incumbent_objs = duplicated_incumbent_obj

    support_env.step(feasible_actions, device)

    tabu_mv = [feasible_actions[torch.argmin(support_env.current_objs, dim=0, keepdim=True).cpu().item()]]

    return tabu_mv


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)  # bug, refer to https://github.com/pytorch/pytorch/issues/61032

    show = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # benchmark config
    l = 1
    h = 99
    init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
    testing_type = ['tai']  # ['syn', 'tai']
    syn_problem_j = [15]
    syn_problem_m = [15]
    # syn_problem_j = [10, 15, 20, 30, 50, 100]
    # syn_problem_m = [10, 15, 20, 20, 20, 20]
    # tai_problem_j = [15]
    # tai_problem_m = [15]
    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]

    # MDP config
    cap_horizon = 2000
    transit = [500, 1000, 2000]  # [500, 1000, 2000]
    result_type = 'incumbent'  # 'current', 'incumbent'
    fea_norm_const = 1000

    for test_t in testing_type:  # select benchmark
        if test_t == 'syn':
            problem_j, problem_m = syn_problem_j, syn_problem_m
        else:
            problem_j, problem_m = tai_problem_j, tai_problem_m

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))
            print('\nStart testing {}{}x{}...'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            if test_t == 'tai':
                gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
            else:
                # ortools solver
                from pathlib import Path
                ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if ortools_path.is_file():
                    gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))[:, 1]
                else:
                    gap_against = []
                    print('Starting Ortools...')
                    for i, data in enumerate(inst):
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        gap_against.append(result[1])
                    gap_against = np.array(gap_against)
                    np.save('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(p_j, p_m), gap_against)

            env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin', fea_norm_const=fea_norm_const)
            support_env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin',
                                 fea_norm_const=fea_norm_const)

            for init in init_type:

                print('Starting rollout Random policy...')
                random_result = []
                random_time = []
                states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev, plot=show)
                random_start = time.time()
                while env.itr < cap_horizon:
                    actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
                    states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
                    for log_horizon in transit:
                        if env.itr == log_horizon:
                            if result_type == 'incumbent':
                                RD_result = env.incumbent_objs.cpu().squeeze().numpy()
                            else:
                                RD_result = env.current_objs.cpu().squeeze().numpy()
                            random_result.append(RD_result)
                            random_time.append(time.time() - random_start)
                random_result = np.array(random_result)
                random_time = np.array(random_time)
                print(random_result.mean(axis=1))
                print(random_time / inst.shape[0])

                print('Starting rollout Best-Improvement policy...')
                best_improvement_result = []
                best_improvement_time = []
                for ins in inst:
                    best_improvement_result_per_instance = []
                    best_improvement_time_per_instance = []
                    ins = np.array([ins])
                    BI_start_per_instance = time.time()
                    _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev, plot=show)
                    while env.itr < cap_horizon:
                        best_actions = best_improvement_move(support_env=support_env,
                                                             feasible_actions=feasible_actions[0],
                                                             current_graph=env.current_graphs[0],
                                                             current_sub_graphs_mc=env.sub_graphs_mc[0],
                                                             current_tabu_list=env.tabu_lists[0],
                                                             current_obj=env.current_objs[0],
                                                             incumbent_obj=env.incumbent_objs[0],
                                                             instance=env.instances[0],
                                                             device=dev)
                        states, _, feasible_actions, _ = env.step(best_actions, dev, plot=show)
                        for log_horizon in transit:
                            if env.itr == log_horizon:
                                if result_type == 'incumbent':
                                    BI_result = env.incumbent_objs.cpu().squeeze().numpy()
                                else:
                                    BI_result = env.current_objs.cpu().squeeze().numpy()
                                best_improvement_result_per_instance.append(BI_result)
                                best_improvement_time_per_instance.append(time.time() - BI_start_per_instance)
                    best_improvement_result.append(best_improvement_result_per_instance)
                    best_improvement_time.append(best_improvement_time_per_instance)
                best_improvement_result = np.array(best_improvement_result)
                best_improvement_time = np.array(best_improvement_time)
                print(best_improvement_result.mean(axis=0))
                print(best_improvement_time.mean(axis=0))

                print('Starting rollout TABU policy...')
                tabu_result = []
                tabu_time = []
                for ins in inst:
                    tabu_result_per_instance = []
                    tabu_time_per_instance = []
                    ins = np.array([ins])
                    TABU_start_per_instance = time.time()
                    _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev, plot=show)
                    while env.itr < cap_horizon:
                        tabu_actions = tabu_move(support_env=support_env,
                                                 feasible_actions=feasible_actions[0],
                                                 current_graph=env.current_graphs[0],
                                                 current_sub_graphs_mc=env.sub_graphs_mc[0],
                                                 current_tabu_list=env.tabu_lists[0],
                                                 current_obj=env.current_objs[0],
                                                 incumbent_obj=env.incumbent_objs[0],
                                                 instance=env.instances[0],
                                                 device=dev)
                        states, _, feasible_actions, _ = env.step(tabu_actions, dev, plot=show)
                        for log_horizon in transit:
                            if env.itr == log_horizon:
                                if result_type == 'incumbent':
                                    tb_result = env.incumbent_objs.cpu().squeeze().numpy()
                                else:
                                    tb_result = env.current_objs.cpu().squeeze().numpy()
                                tabu_result_per_instance.append(tb_result)
                                tabu_time_per_instance.append(time.time() - TABU_start_per_instance)
                    tabu_result.append(tabu_result_per_instance)
                    tabu_time.append(tabu_time_per_instance)
                tabu_result = np.array(tabu_result)
                tabu_time = np.array(tabu_time)
                print(tabu_result.mean(axis=0))
                print(tabu_time.mean(axis=0))


if __name__ == '__main__':
    main()
