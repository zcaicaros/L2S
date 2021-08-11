import numpy as np
import torch
import time
import random
from env.env_batch import JsspN5, BatchGraph
from model.actor import Actor
from ortools_solver import MinimalJobshopSat


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

    # model config
    embedding_type = 'gin'  # 'gin', 'dghan', 'gin+dghan'
    model_init_type = 'fdd-divide-mwkr'
    model_j = 30
    model_m = 20
    heads = 1
    drop_out = 0
    training_episode_length = 500  # [64, 128, 256]
    reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
    model_type = 'incumbent'  # 'incumbent', 'last-step'
    gamma = 1
    hidden_dim = 128
    embedding_layer = 4
    policy_layer = 4
    lr = 5e-5
    steps_learn = 10
    batch_size = 60
    episodes = 128000
    step_validation = 10
    if embedding_type == 'gin':
        dghan_param_for_saved_model = 'NAN'
    elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
        dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
    else:
        raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    # MDP config
    cap_horizon = 2000
    performance_milestones = [500, 1000, 2000]  # [500, 1000, 2000]
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
                .format(model_type, model_j, model_m, l, h, model_init_type, reward_type, gamma,
                        hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
                        lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
            print('loading model from:', saved_model_path)
            policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

            results_each_dateset = []  # save result
            inference_time_each_dataset = []  # save inference time
            for init in init_type:
                results_each_init, inference_time_each_init = [], []
                print('Starting rollout DRL policy...')
                batch_data = BatchGraph()
                states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev, plot=show)
                drl_start = time.time()
                while env.itr < cap_horizon:
                    batch_data.wrapper(*states)
                    actions, _ = policy(batch_data, feasible_actions)
                    states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
                    for log_horizon in performance_milestones:
                        if env.itr == log_horizon:
                            if result_type == 'incumbent':
                                DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                            else:
                                DRL_result = env.current_objs.cpu().squeeze().numpy()
                            print('For testing steps: {}    '.format(env.itr),
                                  'DRL Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                                  'DRL results takes: {:.6f} per instance.'.format((time.time() - drl_start) / inst.shape[0]))
                            results_each_init.append(((DRL_result - gap_against) / gap_against).mean())
                            inference_time_each_init.append((time.time() - drl_start) / inst.shape[0])
                results_each_dateset.append(results_each_init)
                inference_time_each_dataset.append(inference_time_each_init)
            # print(results_each_dateset)
            # print(inference_time_each_dataset)


if __name__ == '__main__':

    main()
