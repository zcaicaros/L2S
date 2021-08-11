import numpy as np
import torch
import time
from env.env_batch import JsspN5
import copy
import random
from ortools_baseline import MinimalJobshopSat


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



show = False
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# benchmark config
l = 1
h = 99
init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
testing_type = ['tai']  # ['syn', 'tai']
syn_problem_j = [10]
syn_problem_m = [10]
# tai_problem_j = [15]
# tai_problem_m = [15]
# syn_problem_j = [10, 15, 20, 30, 50, 100]
# syn_problem_m = [10, 15, 20, 20, 20, 20]
tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]
tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]

# MDP config
transit = [500, 1000, 2000]  # [500, 1000, 2000, 5000, 10000]
result_type = 'incumbent'  # 'current', 'incumbent'
fea_norm_const = 1000



def main():

    for test_t in testing_type:  # select benchmark
        if test_t == 'syn':
            problem_j, problem_m = syn_problem_j, syn_problem_m
        else:
            problem_j, problem_m = tai_problem_j, tai_problem_m

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



            results = []  # save result for DRL and conventional heuristic
            inference_time = []  # save inference for DRL and conventional heuristic

            for init in init_type:
                for test_step in transit:  # select testing max itr
                    results_each_test_step = []
                    inference_time_each_test_step = []
                    env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin', fea_norm_const=fea_norm_const)

                    # saved_actions = []

                    # rollout random policy
                    import random
                    # random.seed(1)
                    print('Starting rollout random policy...')
                    t1_random = time.time()
                    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
                    while env.itr < test_step:
                        actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
                        # print(feasible_actions)
                        # print(actions)
                        # print()
                        # saved_actions.append(actions)
                        states, _, feasible_actions, _ = env.step(actions, dev)
                    if result_type == 'incumbent':
                        Random_result = env.incumbent_objs.cpu().squeeze().numpy()
                    else:
                        Random_result = env.current_objs.cpu().squeeze().numpy()

                    # print(np.array(saved_actions).shape)
                    # np.save('saved_actions.npy', saved_actions)

                    t2_random = time.time()
                    print('Random settings: {}{}x{}, {}, test_step={}'.format(test_t, p_j, p_m, init, test_step))
                    print('Random Gap:', ((Random_result - gap_against) / gap_against).mean())
                    results_each_test_step.append(((Random_result - gap_against) / gap_against).mean())
                    print('Random results takes: {:.4f} per instance.'.format((t2_random - t1_random) / inst.shape[0]))
                    inference_time_each_test_step.append((t2_random - t1_random) / inst.shape[0])
                    # print(Random_result)
                    print()


                    # rollout best_improvement_move
                    # random.seed(1)
                    print('Starting rollout best_improvement_move policy...')
                    support_env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin')
                    best_improvement_result = []
                    t1_best_improvement = time.time()
                    for ins in inst:
                        ins = np.array([ins])
                        _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)
                        while env.itr < test_step:
                            best_move = best_improvement_move(support_env=support_env,
                                                              feasible_actions=feasible_actions[0],
                                                              current_graph=env.current_graphs[0],
                                                              current_tabu_list=env.tabu_lists[0],
                                                              current_obj=env.current_objs[0],
                                                              incumbent_obj=env.incumbent_objs[0],
                                                              instance=env.instances[0],
                                                              device=dev)
                            _, _, feasible_actions, _ = env.step(best_move, dev)
                        if result_type == 'incumbent':
                            best_improvement_result.append(env.incumbent_objs.cpu().item())
                        else:
                            best_improvement_result.append(env.current_objs.cpu().item())
                    t2_best_improvement = time.time()
                    best_improvement_result = np.array(best_improvement_result)
                    print('Best_improvement_move settings: {}{}x{}, {}, test_step={}'.format(test_t, p_j, p_m, init, test_step))
                    print('Best_improvement_move Gap:', ((best_improvement_result - gap_against) / gap_against).mean())
                    results_each_test_step.append(((best_improvement_result - gap_against) / gap_against).mean())
                    print('Best_improvement_move results takes: {:.4f} per instance.'.format(
                        (t2_best_improvement - t1_best_improvement) / inst.shape[0]))
                    inference_time_each_test_step.append((t2_best_improvement - t1_best_improvement) / inst.shape[0])
                    # print(best_improvement_result)
                    print()
                    results.append(results_each_test_step)
                    inference_time.append(inference_time_each_test_step)

                    # rollout tabu_move
                    # random.seed(1)
                    print('Starting rollout tabu_move policy...')
                    support_env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin')
                    tabu_result = []
                    t1_tabu = time.time()
                    for ins in inst:
                        ins = np.array([ins])
                        _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)
                        while env.itr < test_step:
                            best_move = tabu_move(support_env=support_env,
                                                  feasible_actions=feasible_actions[0],
                                                  current_graph=env.current_graphs[0],
                                                  current_tabu_list=env.tabu_lists[0],
                                                  current_obj=env.current_objs[0],
                                                  incumbent_obj=env.incumbent_objs[0],
                                                  instance=env.instances[0],
                                                  device=dev)
                            _, _, feasible_actions, _ = env.step(best_move, dev)
                        if result_type == 'incumbent':
                            tabu_result.append(env.incumbent_objs.cpu().item())
                        else:
                            tabu_result.append(env.current_objs.cpu().item())
                    t2_tabu = time.time()
                    tabu_result = np.array(tabu_result)
                    print('Tabu_move settings: {}{}x{}, {}, test_step={}'.format(test_t, p_j, p_m, init, test_step))
                    print('Tabu_move Gap:', ((tabu_result - gap_against) / gap_against).mean())
                    results_each_test_step.append(((tabu_result - gap_against) / gap_against).mean())
                    print('Tabu_move results takes: {:.4f} per instance.'.format(
                        (t2_tabu - t1_tabu) / inst.shape[0]))
                    inference_time_each_test_step.append((t2_tabu - t1_tabu) / inst.shape[0])
                    # print(best_improvement_result)
                    print()
                    results.append(results_each_test_step)
                    inference_time.append(inference_time_each_test_step)

            # np.save('test_results/results_{}{}x{}.npy'.format(test_t, p_j, p_m), np.array(results))
            # np.save('test_results/inference_time_{}{}x{}.npy'.format(test_t, p_j, p_m), np.array(inference_time))


if __name__ == '__main__':
    # seed
    random.seed(1)
    main()