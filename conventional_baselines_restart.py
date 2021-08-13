import numpy as np
import torch
import time
from env.env_batch import JsspN5
import copy
import random
from ortools_solver import MinimalJobshopSat


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
        best_move = [[0, 0]]

    return best_move


def first_improvement_move(support_env,
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
        first_improved_idx = torch.nonzero(support_env.current_objs < current_obj)[0][0].cpu().item()
        best_move = [feasible_actions[first_improved_idx]]
    else:
        best_move = [[0, 0]]

    return best_move


def greedy_move(support_env,
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

    greedy_mv = [feasible_actions[torch.argmin(support_env.current_objs, dim=0, keepdim=True).cpu().item()]]

    return greedy_mv


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
    testing_type = ['syn', 'tai', 'abz', 'orb', 'yn', 'swv', 'la']  # ['syn', 'tai', 'abz', 'orb', 'yn', 'swv', 'la']
    # syn_problem_j = [15]
    # syn_problem_m = [15]
    syn_problem_j = [10, 15, 20, 30]  # [10, 15, 20, 30, 50, 100]
    syn_problem_m = [10, 15, 20, 20]  # [10, 15, 20, 20, 20, 20]
    # tai_problem_j = [15]
    # tai_problem_m = [15]
    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]
    abz_problem_j = [10, 20]
    abz_problem_m = [10, 15]
    orb_problem_j = [10]
    orb_problem_m = [10]
    yn_problem_j = [20]
    yn_problem_m = [20]
    swv_problem_j = [20, 20, 50]
    swv_problem_m = [10, 15, 10]
    la_problem_j = [10, 15, 20, 10, 15, 20, 30]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10]

    # MDP config
    cap_horizon = 5000
    transit = [500, 1000, 2000, 5000]  # [500, 1000, 2000]
    result_type = 'incumbent'  # 'current', 'incumbent'
    fea_norm_const = 1000

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
        else:
            raise Exception('Problem type must be in testing_type = ["syn", "tai", "abz", "orb", "yn", "swv", "la"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))
            print('\nStart testing {}{}x{}...\n'.format(test_t, p_j, p_m))

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

                gap_against_tiled = np.tile(gap_against, (len(transit), 1))

                print('Starting rollout Greedy policy...')
                result_greedy = []
                time_greedy = []
                for ins in inst:
                    greedy_result_per_instance = []
                    greedy_time_per_instance = []
                    ins = np.array([ins])
                    greedy_per_instance_start_time = time.time()
                    _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev, plot=show)
                    while env.itr < cap_horizon:
                        greedy_actions = greedy_move(support_env=support_env,
                                                     feasible_actions=feasible_actions[0],
                                                     current_graph=env.current_graphs[0],
                                                     current_sub_graphs_mc=env.sub_graphs_mc[0],
                                                     current_tabu_list=env.tabu_lists[0],
                                                     current_obj=env.current_objs[0],
                                                     incumbent_obj=env.incumbent_objs[0],
                                                     instance=env.instances[0],
                                                     device=dev)
                        _, _, feasible_actions, _ = env.step(greedy_actions, dev, plot=show)
                        for log_horizon in transit:
                            if env.itr == log_horizon:
                                if result_type == 'incumbent':
                                    greedy_result = env.incumbent_objs.cpu().squeeze().numpy()
                                else:
                                    greedy_result = env.current_objs.cpu().squeeze().numpy()
                                greedy_result_per_instance.append(greedy_result)
                                greedy_time_per_instance.append(time.time() - greedy_per_instance_start_time)
                    result_greedy.append(greedy_result_per_instance)
                    time_greedy.append(greedy_time_per_instance)
                result_greedy = np.array(result_greedy).transpose()
                time_greedy = np.array(time_greedy).transpose()
                gap_greedy = (result_greedy - gap_against_tiled) / gap_against_tiled
                mean_gap_greedy = gap_greedy.mean(axis=1)
                mean_time_greedy = time_greedy.mean(axis=1)
                print('Averaged gap for greedy policy: ', mean_gap_greedy)
                print('Averaged time for greedy policy: ', mean_time_greedy)

                print()

                print('Starting rollout Best-Improvement policy...')
                result_best_improvement = []
                time_best_improvement = []
                for ins in inst[:1]:
                    results_with_restart_per_instance = []
                    best_improvement_result_per_instance = []
                    best_improvement_time_per_instance = []
                    ins = np.array([ins])
                    BI_start_per_instance = time.time()
                    _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev, plot=show)
                    steps_count = 0
                    while steps_count < cap_horizon:
                        best_actions = best_improvement_move(support_env=support_env,
                                                             feasible_actions=feasible_actions[0],
                                                             current_graph=env.current_graphs[0],
                                                             current_sub_graphs_mc=env.sub_graphs_mc[0],
                                                             current_tabu_list=env.tabu_lists[0],
                                                             current_obj=env.current_objs[0],
                                                             incumbent_obj=env.incumbent_objs[0],
                                                             instance=env.instances[0],
                                                             device=dev)
                        if best_actions == [[0, 0]]:
                            if result_type == 'incumbent':
                                results_with_restart_per_instance.append(env.incumbent_objs.cpu().item())
                            else:
                                results_with_restart_per_instance.append(env.current_objs.cpu().item())
                            _, feasible_actions, _ = env.reset(instances=ins, init_type='plist', device=dev, plot=show)
                        else:
                            _, _, feasible_actions, _ = env.step(best_actions, dev, plot=show)
                        steps_count += 1
                        for log_horizon in transit:
                            if steps_count == log_horizon:
                                best_improvement_result_per_instance.append(min(results_with_restart_per_instance))
                                best_improvement_time_per_instance.append(time.time() - BI_start_per_instance)
                    result_best_improvement.append(best_improvement_result_per_instance)
                    time_best_improvement.append(best_improvement_time_per_instance)
                result_best_improvement = np.array(result_best_improvement).transpose()
                time_best_improvement = np.array(time_best_improvement).transpose()
                gap_best_improvement = (result_best_improvement - gap_against_tiled) / gap_against_tiled
                mean_gap_best_improvement = gap_best_improvement.mean(axis=1)
                mean_time_best_improvement = time_best_improvement.mean(axis=1)
                print('Averaged gap for best improvement policy: ', mean_gap_best_improvement)
                print('Averaged time for best improvement policy: ', mean_time_best_improvement)

                print()

                print('Starting rollout First-Improvement policy...')
                result_first_improvement = []
                time_first_improvement = []
                for ins in inst[:1]:
                    results_with_restart_per_instance = []
                    first_improvement_result_per_instance = []
                    first_improvement_time_per_instance = []
                    ins = np.array([ins])
                    FRSTI_start_per_instance = time.time()
                    _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev, plot=show)
                    steps_count = 0
                    while steps_count < cap_horizon:
                        first_improved_actions = first_improvement_move(support_env=support_env,
                                                                        feasible_actions=feasible_actions[0],
                                                                        current_graph=env.current_graphs[0],
                                                                        current_sub_graphs_mc=env.sub_graphs_mc[0],
                                                                        current_tabu_list=env.tabu_lists[0],
                                                                        current_obj=env.current_objs[0],
                                                                        incumbent_obj=env.incumbent_objs[0],
                                                                        instance=env.instances[0],
                                                                        device=dev)
                        if first_improved_actions == [[0, 0]]:
                            if result_type == 'incumbent':
                                results_with_restart_per_instance.append(env.incumbent_objs.cpu().item())
                            else:
                                results_with_restart_per_instance.append(env.current_objs.cpu().item())
                            _, feasible_actions, _ = env.reset(instances=ins, init_type='plist', device=dev, plot=show)
                        else:
                            _, _, feasible_actions, _ = env.step(first_improved_actions, dev, plot=show)
                        steps_count += 1
                        for log_horizon in transit:
                            if steps_count == log_horizon:
                                first_improvement_result_per_instance.append(min(results_with_restart_per_instance))
                                first_improvement_time_per_instance.append(time.time() - FRSTI_start_per_instance)
                    result_first_improvement.append(first_improvement_result_per_instance)
                    time_first_improvement.append(first_improvement_time_per_instance)
                result_first_improvement = np.array(result_first_improvement).transpose()
                time_first_improvement = np.array(time_first_improvement).transpose()
                gap_first_improvement = (result_first_improvement - gap_against_tiled) / gap_against_tiled
                mean_gap_first_improvement = gap_first_improvement.mean(axis=1)
                mean_time_first_improvement = time_first_improvement.mean(axis=1)
                print('Averaged gap for first improvement policy: ', mean_gap_first_improvement)
                print('Averaged time for first improvement policy: ', mean_time_first_improvement)

                '''print('Starting rollout Random policy...')
                                result_random = []
                                time_random = []
                                _, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev, plot=show)
                                start_random = time.time()
                                while env.itr < cap_horizon:
                                    actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
                                    _, _, feasible_actions, _ = env.step(actions, dev, plot=show)
                                    for log_horizon in transit:
                                        if env.itr == log_horizon:
                                            if result_type == 'incumbent':
                                                result_RD = env.incumbent_objs.cpu().squeeze().numpy()
                                            else:
                                                result_RD = env.current_objs.cpu().squeeze().numpy()
                                            result_random.append(result_RD)
                                            time_random.append(time.time() - start_random)
                                result_random = np.array(result_random)
                                time_random = np.array(time_random)
                                gap_random = (result_random - gap_against_tiled) / gap_against_tiled
                                mean_gap_random = gap_random.mean(axis=1)
                                mean_time_random = time_random / inst.shape[0]
                                print('Averaged gap for random policy: ', mean_gap_random)
                                print('Averaged time for random policy: ', mean_time_random)'''


if __name__ == '__main__':
    main()
