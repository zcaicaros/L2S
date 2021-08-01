import numpy as np
import torch
import time
import copy
import random
from env.env_batch import JsspN5
from model.actor import Actor
from ortools_baseline import MinimalJobshopSat
from env.env_batch import BatchGraph
from conventional_baselines import best_improvement_move

show = False
# torch.manual_seed(1)
# np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# benchmark config
l = 1
h = 99
init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
testing_type = ['syn', 'tai']  # ['syn', 'tai']
syn_problem_j = [10]
syn_problem_m = [10]
# tai_problem_j = [15]
# tai_problem_m = [15]
# syn_problem_j = [10, 15, 20, 30, 50, 100]
# syn_problem_m = [10, 15, 20, 20, 20, 20]
tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]
tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]

# model config
model_j = [10]
model_m = [10]
training_episode_length = [500]  # [64, 128, 256]
reward_type = ['yaoxin']  # ['yaoxin', 'consecutive']
model_type = ['incumbent']  # ['incumbent', 'last-step']
embedding_type = ['gin']  # ['gin', 'dghan', 'gin+dghan']
gamma = 1
hidden_dim = 128
embedding_layer = 4
policy_layer = 4
lr = 5e-5
steps_learn = 10
batch_size = 256
episodes = 128000
step_validation = 10


# MDP config
transit = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000, 10000]
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
                    print('Starting rollout DRL policy...')
                    for embd_type in embedding_type:
                        policy = Actor(3, 128, gin_l=4, policy_l=4, embedding_type=embd_type).to(dev)
                        for r_type in reward_type:  # select reward type
                            for training_length in training_episode_length:  # select training episode length
                                for m_j, m_m in zip(model_j, model_m):  # select training model size
                                    for m_type in model_type:  # select training model type
                                        torch.manual_seed(1)
                                        saved_model_path = './renamed_saved_model/' \
                                                           '{}_{}x{}[{},{}]_{}_{}_{}_' \
                                                           '{}_{}_{}_{}_' \
                                                           '{}_{}_{}_{}_{}_{}' \
                                                           '.pth'\
                                            .format(m_type, m_j, m_m, l, h, init, r_type, gamma,
                                                    hidden_dim, embedding_layer, policy_layer, embd_type,
                                                    lr, steps_learn, training_length, batch_size, episodes, step_validation)
                                        print('loading model from:', saved_model_path)
                                        policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))
                                        batch_data = BatchGraph()
                                        # rollout network
                                        t1_drl = time.time()
                                        states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
                                        while env.itr < test_step:
                                            batch_data.wrapper(*states)
                                            actions, _ = policy(batch_data, feasible_actions)
                                            states, _, feasible_actions, _ = env.step(actions, dev)
                                        if result_type == 'incumbent':
                                            DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                                        else:
                                            DRL_result = env.current_objs.cpu().squeeze().numpy()
                                        t2_drl = time.time()
                                        print(
                                            'DRL settings: {}{}x{}, {}, test_step={}, reward_type={}, model_type={}, model_training_length={}'.format(
                                                test_t, p_j, p_m, init, test_step, r_type, m_type, training_length))
                                        print('DRL Gap:', ((DRL_result - gap_against) / gap_against).mean())
                                        results_each_test_step.append(((DRL_result - gap_against) / gap_against).mean())
                                        print(
                                            'DRL results takes: {:.4f} per instance.'.format((t2_drl - t1_drl) / inst.shape[0]))
                                        inference_time_each_test_step.append((t2_drl - t1_drl) / inst.shape[0])
                                        # print(DRL_result)
                                        print()


                    # rollout random policy
                    import random
                    random.seed(1)
                    print('Starting rollout random policy...')
                    t1_random = time.time()
                    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
                    while env.itr < test_step:
                        actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
                        states, _, feasible_actions, _ = env.step(actions, dev)
                    if result_type == 'incumbent':
                        Random_result = env.incumbent_objs.cpu().squeeze().numpy()
                    else:
                        Random_result = env.current_objs.cpu().squeeze().numpy()

                    t2_random = time.time()
                    print('Random settings: {}{}x{}, {}, test_step={}'.format(test_t, p_j, p_m, init, test_step))
                    print('Random Gap:', ((Random_result - gap_against) / gap_against).mean())
                    results_each_test_step.append(((Random_result - gap_against) / gap_against).mean())
                    print('Random results takes: {:.4f} per instance.'.format((t2_random - t1_random) / inst.shape[0]))
                    inference_time_each_test_step.append((t2_random - t1_random) / inst.shape[0])
                    # print(Random_result)
                    print()


                    # rollout greedy
                    random.seed(1)
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

            np.save('testing_results/results_{}{}x{}.npy'.format(test_t, p_j, p_m), np.array(results))
            np.save('testing_results/inference_time_{}{}x{}.npy'.format(test_t, p_j, p_m), np.array(inference_time))


if __name__ == '__main__':
    main()
