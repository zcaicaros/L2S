import numpy as np
import os
import torch
import time
import random
from env.environment import JsspN5, BatchGraph
from model.actor import Actor
from ortools_solver import MinimalJobshopSat


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)  # bug, refer to https://github.com/pytorch/pytorch/issues/61032

    show = False
    force_test = True
    save = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using cuda to test...')

    # benchmark config
    p_l = 1
    p_h = 99
    init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
    testing_type = ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft']  # ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn']
    syn_problem_j = [10, 15, 15, 20, 20, 100, 150]  # [10, 15, 15, 20, 20, 100, 150]
    syn_problem_m = [10, 10, 15, 10, 15, 20, 25]  # [10, 10, 15, 10, 15, 20, 25]
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
    la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]
    ft_problem_j = [6, 10, 20]  # [6, 10, 20]
    ft_problem_m = [6, 10, 5]  # [6, 10, 5]

    # model config
    model_j = 10  # 10， 15， 15， 20， 20
    model_m = 10  # 10， 10， 15， 10， 15
    model_l = 1
    model_h = 99
    model_init_type = 'fdd-divide-mwkr'
    reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
    gamma = 1

    hidden_dim = 128
    embedding_layer = 4
    policy_layer = 4
    embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
    heads = 1
    drop_out = 0.

    lr = 5e-5  # 5e-5, 4e-5
    steps_learn = 10
    training_episode_length = 500
    batch_size = 64
    episodes = 128000  # 128000, 256000
    step_validation = 10

    model_type = 'incumbent'  # 'incumbent', 'last-step'

    if embedding_type == 'gin':
        dghan_param_for_saved_model = 'NAN'
    elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
        dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
    else:
        raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    # MDP config
    cap_horizon = 5000
    performance_milestones = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]
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
        elif test_t == 'ft':
            problem_j, problem_m = ft_problem_j, ft_problem_m
        else:
            raise Exception(
                'Problem type must be in testing_type = ["tai", "abz", "orb", "yn", "swv", "la", "ft", "syn"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))
            print('\nStart testing {}{}x{}...'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            if test_t != 'syn':
                gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
            else:
                # ortools solver
                from pathlib import Path
                ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if ortools_path.is_file():
                    gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                else:
                    ortools_results = []
                    print('Starting Ortools...')
                    for i, data in enumerate(inst):
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        ortools_results.append(result)
                    ortools_results = np.array(ortools_results)
                    np.save('./test_data/syn{}x{}_result.npy'.format(p_j, p_m), ortools_results)
                    gap_against = ortools_results[:, 1]

            env = JsspN5(n_job=p_j, n_mch=p_m, low=p_l, high=p_h, reward_type='yaoxin', fea_norm_const=fea_norm_const)
            torch.manual_seed(seed)
            policy = Actor(in_dim=3,
                           hidden_dim=hidden_dim,
                           embedding_l=embedding_layer,
                           policy_l=policy_layer,
                           embedding_type=embedding_type,
                           heads=heads,
                           dropout=drop_out).to(dev)
            saved_model_path = './saved_model/' \
                               '{}_{}x{}[{},{}]_{}_{}_{}_' \
                               '{}_{}_{}_{}_{}_' \
                               '{}_{}_{}_{}_{}_{}' \
                               '.pth' \
                .format(model_type, model_j, model_m, model_l, model_h, model_init_type, reward_type, gamma,
                        hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                        lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
            print('loading model from:', saved_model_path)
            policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

            for init in init_type:
                from pathlib import Path
                print('Start to test initial solution: {}...'.format(init))
                which_model = './test_results/DRL_results/' \
                              '{}_{}x{}[{},{}]_{}_{}_{}_' \
                              '{}_{}_{}_{}_{}_' \
                              '{}_{}_{}_{}_{}_{}/' \
                    .format(model_type, model_j, model_m, model_l, model_h, model_init_type, reward_type, gamma,
                            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
                which_dateset = '{}_{}x{}_{}'.format(test_t, p_j, p_m, init)
                if not Path(which_model).is_dir():
                    os.mkdir(which_model)
                    with open(which_model + '__init__.py', 'w'): pass  # make created folder as python package by creating __init__.py
                result_file = Path(which_model + which_dateset + '_result.npy')
                time_file = Path(which_model + which_dateset + '_time.npy')
                if not result_file.is_file() or not time_file.is_file():
                    print('Starting rollout DRL policy...')
                    if p_j >= 100 and inst.shape[0] >= 20:
                        chunk_size = 20
                        print('Problem of size {}x{} containing {} instances is too large to form a batch. Splitting into chunks and test seperately. Chunk size is {}.'.format(p_j, p_m, inst.shape[0], chunk_size))
                    else:
                        chunk_size = inst.shape[0]
                    n_chunks = inst.shape[0] // chunk_size
                    results_each_init, inference_time_each_init = [], []
                    for i in range(n_chunks):
                        # t3 = time.time()
                        chunk_result, chunk_time = [], []
                        inst_chunk = inst[i*chunk_size:(i+1)*chunk_size]
                        batch_data = BatchGraph()
                        states, feasible_actions, _ = env.reset(instances=inst_chunk, init_type=init, device=dev, plot=show)
                        # t4 = time.time()
                        drl_start = time.time()
                        while env.itr < cap_horizon:
                            # t1 = time.time()
                            batch_data.wrapper(*states)
                            actions, _ = policy(batch_data, feasible_actions)
                            states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
                            # t2 = time.time()
                            for log_horizon in performance_milestones:
                                if env.itr == log_horizon:
                                    if result_type == 'incumbent':
                                        DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                                    else:
                                        DRL_result = env.current_objs.cpu().squeeze().numpy()
                                    chunk_result.append(DRL_result)
                                    chunk_time.append(time.time() - drl_start)
                                    if n_chunks == 1:
                                        print('For testing steps: {}    '.format(env.itr),
                                              'DRL Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                                              'DRL results takes: {:.6f} per instance.'.format((time.time() - drl_start) / chunk_size))
                        results_each_init.append(np.stack(chunk_result))
                        inference_time_each_init.append(np.array(chunk_time))
                    results_each_init = np.concatenate(results_each_init, axis=-1)
                    inference_time_each_init = ((np.stack(inference_time_each_init).sum(axis=0)) / n_chunks) / chunk_size
                    if n_chunks > 1:
                        for i, step in enumerate(performance_milestones):
                            print('For testing steps: {}    '.format(step),
                                  'DRL Gap: {:.6f}    '.format(((results_each_init[i] - gap_against) / gap_against).mean()),
                                  'DRL results takes: {:.6f} per instance.'.format(inference_time_each_init[i]))
                    if save:
                        np.save(which_model + which_dateset + '_result.npy', results_each_init)
                        np.save(which_model + which_dateset + '_time.npy', inference_time_each_init)
                else:
                    if force_test:
                        print('Starting rollout DRL policy...')
                        if p_j >= 100 and inst.shape[0] >= 20:
                            chunk_size = 20
                            print(
                                'Problem of size {}x{} containing {} instances is too large to form a batch. Splitting into chunks and test seperately. Chunk size is {}.'.format(
                                    p_j, p_m, inst.shape[0], chunk_size))
                        else:
                            chunk_size = inst.shape[0]
                        n_chunks = inst.shape[0] // chunk_size
                        results_each_init, inference_time_each_init = [], []
                        for i in range(n_chunks):
                            # t3 = time.time()
                            chunk_result, chunk_time = [], []
                            inst_chunk = inst[i * chunk_size:(i + 1) * chunk_size]
                            batch_data = BatchGraph()
                            states, feasible_actions, _ = env.reset(instances=inst_chunk, init_type=init, device=dev,
                                                                    plot=show)
                            # t4 = time.time()
                            drl_start = time.time()
                            while env.itr < cap_horizon:
                                # t1 = time.time()
                                batch_data.wrapper(*states)
                                actions, _ = policy(batch_data, feasible_actions)
                                states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
                                # t2 = time.time()
                                for log_horizon in performance_milestones:
                                    if env.itr == log_horizon:
                                        if result_type == 'incumbent':
                                            DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                                        else:
                                            DRL_result = env.current_objs.cpu().squeeze().numpy()
                                        chunk_result.append(DRL_result)
                                        chunk_time.append(time.time() - drl_start)
                                        if n_chunks == 1:
                                            print('For testing steps: {}    '.format(env.itr),
                                                  'DRL Gap: {:.6f}    '.format(
                                                      ((DRL_result - gap_against) / gap_against).mean()),
                                                  'DRL results takes: {:.6f} per instance.'.format(
                                                      (time.time() - drl_start) / chunk_size))
                            results_each_init.append(np.stack(chunk_result))
                            inference_time_each_init.append(np.array(chunk_time))
                        results_each_init = np.concatenate(results_each_init, axis=-1)
                        inference_time_each_init = ((np.stack(inference_time_each_init).sum(
                            axis=0)) / n_chunks) / chunk_size
                        if n_chunks > 1:
                            for i, step in enumerate(performance_milestones):
                                print('For testing steps: {}    '.format(step),
                                      'DRL Gap: {:.6f}    '.format(
                                          ((results_each_init[i] - gap_against) / gap_against).mean()),
                                      'DRL results takes: {:.6f} per instance.'.format(inference_time_each_init[i]))
                        if save:
                            np.save(which_model + which_dateset + '_result.npy', results_each_init)
                            np.save(which_model + which_dateset + '_time.npy', inference_time_each_init)
                    else:
                        print('Results already exist.')


if __name__ == '__main__':

    main()
