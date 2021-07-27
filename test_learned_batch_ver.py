import numpy as np
import torch
import time
import copy
from env.env_batch import JsspN5
from model.actor import Actor
from ortools_baseline import MinimalJobshopSat
from env.env_batch import BatchGraph


show = False

# problem config
p_j = 10
p_m = 10
l = 1
h = 99
testing_type = 'syn'  # 'tai', 'syn'
n_generated_instances = 100
init = 'fdd-divide-mwkr'  # 'fdd-divide-mwkr', 'spt', ...

# model config
model_j = 10
model_m = 10

training_episode_length = [64, 128, 256]
reward_type = ['yaoxin', 'consecutive']  # 'yaoxin', 'consecutive'
model_type = ['incumbent', 'current']  # 'incumbent', 'current'

# MDP config
transit = [500, 1000, 2000]
result_type = 'incumbent'  # 'current', 'incumbent'

torch.manual_seed(1)
np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def best_improvement_move(feasible_actions, current_graph, current_tabu_list, current_obj, incumbent_obj, instance, device):

    # only support single instance, so env.inst.shape = [b=1, 2, j, m]

    n_feasible_actions = len(feasible_actions)

    duplicated_instances = np.tile(instance, reps=[n_feasible_actions, 1, 1, 1])
    duplicated_current_obj = current_obj.repeat(n_feasible_actions, 1)
    duplicated_incumbent_obj = incumbent_obj.repeat(n_feasible_actions, 1)
    duplicated_current_graphs = [copy.deepcopy(current_graph) for _ in range(n_feasible_actions)]
    duplicated_tabu_lists = [copy.copy(current_tabu_list) for _ in range(n_feasible_actions)]

    support_env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type=reward_type)
    support_env.instances = duplicated_instances
    support_env.current_graphs = duplicated_current_graphs
    support_env.current_objs = duplicated_current_obj
    support_env.tabu_lists = duplicated_tabu_lists
    support_env.incumbent_objs = duplicated_incumbent_obj

    support_env.step(feasible_actions, device)

    if support_env.current_objs.min().cpu().item() < current_obj.cpu().item():
        best_move = [feasible_actions[torch.argmin(support_env.current_objs, dim=0, keepdim=True).cpu().item()]]
    else:
        best_move = [[0, 0]]  # if there is no better move, don't move

    return best_move


def main():

    if testing_type == 'tai':
        inst = np.load('./test_data/tai{}x{}.npy'.format(p_j, p_m))
    elif testing_type == 'syn':
        inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m))
    else:
        inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m))

    if testing_type == 'tai':
        tai_sota_result = np.load('./test_data/tai{}x{}_SOTA_result.npy'.format(p_j, p_m))
        gap_against = tai_sota_result
    else:
        # ortools solver
        from pathlib import Path
        ortools_path = Path('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(p_j, p_m))
        if ortools_path.is_file():
            results_ortools = np.load('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(p_j, p_m))
        else:
            results_ortools = []
            print('Starting Ortools...')
            for i, data in enumerate(inst):
                times_rearrange = np.expand_dims(data[0], axis=-1)
                machines_rearrange = np.expand_dims(data[1], axis=-1)
                data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                result = MinimalJobshopSat(data.tolist())
                print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                results_ortools.append(result[1])
            results_ortools = np.array(results_ortools)
            np.save('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(p_j, p_m), results_ortools)
        gap_against = results_ortools

    policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)

    for test_step in transit:
        env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type='yaoxin')
        for r_type in reward_type:
            for training_length in training_episode_length:
                for m_type in model_type:
                    # torch.manual_seed(1)
                    saved_model_path = './saved_model/{}x{}_{}_{}_{}_{}_reward.pth'.format(model_j, model_m, init,
                                                                                           training_length,
                                                                                           m_type, r_type)
                    print(saved_model_path)
                    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))
                    batch_data = BatchGraph()
                    # rollout network
                    print('Starting rollout DRL policy...')
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
                    print('DRL settings: test_step={}, reward_type={}, model_type={}, model_training_length={}'.format(test_step, r_type, m_type, training_length))
                    print('DRL Gap:', ((DRL_result - gap_against) / gap_against).mean())
                    print('DRL results takes: {:.4f} per instance.'.format((t2_drl - t1_drl)/inst.shape[0]))
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
        print('Random settings: test_step={}'.format(test_step))
        print('Random Gap:', ((Random_result - gap_against) / gap_against).mean())
        print('Random results takes: {:.4f} per instance.'.format((t2_random - t1_random)/inst.shape[0]))
        # print(Random_result)
        print()

        # print('DRL improves {0:.2%} against Random'.format(((DRL_result - Random_result)/Random_result).mean()))

        # rollout greedy
        print('Starting rollout greedy policy...')
        t1_best_improvement = time.time()
        best_improvement_result = []
        for ins in inst:
            ins = np.array([ins])
            _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)
            last_obj = env.incumbent_objs.cpu().item()
            while env.itr < test_step:
                s_step = time.time()
                best_move = best_improvement_move(feasible_actions=feasible_actions[0],
                                                  current_graph=env.current_graphs[0],
                                                  current_tabu_list=env.tabu_lists[0],
                                                  current_obj=env.current_objs[0],
                                                  incumbent_obj=env.incumbent_objs[0],
                                                  instance=env.instances[0],
                                                  device=dev)
                _, _, feasible_actions, _ = env.step(best_move, dev)
                t_step = time.time()
                if last_obj == env.incumbent_objs.cpu().item():
                    break
                last_obj = env.incumbent_objs.cpu().item()
            best_improvement_result.append(env.incumbent_objs.cpu().item())
        t2_best_improvement = time.time()
        best_improvement_result = np.array(best_improvement_result)
        print('Greedy settings: test_step={}'.format(test_step))
        print('Greedy Gap:', ((best_improvement_result - gap_against) / gap_against).mean())
        print('Greedy results takes: {:.4f} per instance.'.format((t_step - s_step)*test_step))
        # print(best_improvement_result)
        print()


if __name__ == '__main__':
    main()