import numpy as np
import torch
import time
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

# model config
model_j = 10
model_m = 10
training_episode_length = 128
init = 'fdd-divide-mwkr'  # 'fdd-divide-mwkr', 'spt', ...
model_type = 'current'  # 'current', 'incumbent'
reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'

# MDP config
transit = 2000


torch.manual_seed(1)
np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type=reward_type)
    policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)
    saved_model_path = './saved_model/{}x{}_{}_{}_{}_{}_reward.pth'.format(model_j, model_m, init, training_episode_length, model_type, reward_type)
    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))
    batch_data = BatchGraph()

    # inst = np.array([uni_instance_gen(n_j=p_j, n_m=p_m, low=l, high=h) for _ in range(n_generated_instances)])
    # np.save('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m), inst)

    if testing_type == 'tai':
        inst = np.load('./test_data/tai{}x{}.npy'.format(p_j, p_m))
    elif testing_type == 'syn':
        inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m))
    else:
        inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m))

    # rollout network
    print('Starting rollout DRL policy...')
    t1_drl = time.time()
    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
    while env.itr < transit:
        batch_data.wrapper(*states)
        actions, _ = policy(batch_data, feasible_actions)
        states, _, feasible_actions, _ = env.step(actions, dev)
    DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
    t2_drl = time.time()
    print('DRL results takes: {:.4f}s per instance.\n'.format((t2_drl - t1_drl)/inst.shape[0]), DRL_result)

    # rollout random policy
    import random
    random.seed(1)
    print('Starting rollout random policy...')
    t1_random = time.time()
    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
    while env.itr < transit:
        actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
        states, _, feasible_actions, _ = env.step(actions, dev)
    Random_result = env.incumbent_objs.cpu().squeeze().numpy()
    t2_random = time.time()
    print('Random results takes: {:.4f}s per instance.\n'.format((t2_random - t1_random)/inst.shape[0]), Random_result)

    # print('DRL improves {0:.2%} against Random'.format(((DRL_result - Random_result)/Random_result).mean()))

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


    print('DRL Gap:', ((DRL_result - gap_against) / gap_against).mean())
    print('Random Gap:', ((Random_result - gap_against) / gap_against).mean())


if __name__ == '__main__':
    main()