import numpy as np
import os
import torch
import time
import random
from env.environment import JsspN5, BatchGraph
from model.actor import Actor
from env.generateJSP import uni_instance_gen


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)  # bug, refer to https://github.com/pytorch/pytorch/issues/61032

    show = False
    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = 'cpu'
    print('\nusing {} to test...'.format(dev))

    p_l = 1
    p_h = 99
    init_type = 'fdd-divide-mwkr'
    fixed_j = 30
    problem_m = [5, 10, 15, 20, 25, 30]
    problem_j = [fixed_j for _ in range(len(problem_m))]
    # fixed_m = 5
    # problem_j = [5, 10, 15, 20, 25, 30]
    # problem_m = [fixed_m for _ in range(len(problem_j))]
    instance_batch_size = 1

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
    embedding_type = 'gin+dghan'
    heads = 1
    drop_out = 0.

    lr = 5e-5  # 5e-5, 4e-5
    steps_learn = 10
    training_episode_length = 500
    batch_size = 64
    episodes = 128000  # 128000, 256000
    step_validation = 10

    model_type = 'incumbent'  # 'incumbent', 'last-step'

    dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)

    # MDP config
    cap_horizon = 1500
    performance_milestones = [500, 1000, 1500]  # [500, 1000, 2000, 5000]
    fea_norm_const = 1000


    times = []
    for p_j, p_m in zip(problem_j, problem_m):  # select problem size

        times_each_size = []

        inst = np.array([uni_instance_gen(p_j, p_m, p_l, p_h) for _ in range(instance_batch_size)])
        print('\nStart testing {}x{}...'.format(p_j, p_m))

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

        print('Starting rollout DRL policy...')
        batch_data = BatchGraph()
        states, feasible_actions, _ = env.reset(instances=inst, init_type=init_type, device=dev, plot=show)
        drl_start = time.time()
        while env.itr < cap_horizon:
            batch_data.wrapper(*states)
            actions, _ = policy(batch_data, feasible_actions)
            states, _, feasible_actions, _ = env.step(actions, dev, plot=show)
            for log_horizon in performance_milestones:
                if env.itr == log_horizon:
                    times_each_size.append((time.time() - drl_start) / inst.shape[0])
                    print('For testing steps: {}    '.format(env.itr),
                          'DRL results takes: {:.6f} per instance.'.format((time.time() - drl_start) / inst.shape[0]))

        times.append(times_each_size)

    times = np.array(times)
    np.save('./complexity/L2S_complexity_fixed_j={}_{}.npy'.format(fixed_j, performance_milestones), times)
    # np.save('./complexity/L2S_complexity_fixed_m={}_{}.npy'.format(fixed_m, performance_milestones), times)


if __name__ == '__main__':

    main()
