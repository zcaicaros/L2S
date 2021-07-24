import numpy as np
import torch
import time
from env.env_batch import JsspN5
from model.actor import Actor
from ortools_baseline import MinimalJobshopSat
from env.env_batch import BatchGraph


show = True
j = 10
m = 10
l = 1
h = 99
episode_length = 128
n_generated_instances = 100
transit = 500
init = 'fdd-divide-mwkr'  # 'plist', 'spt', ...
model_type = 'current'

torch.manual_seed(1)
np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    saved_model_path = './saved_model/{}x{}_{}_{}_{}.pth'.format(j, m, init, episode_length, model_type)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h)
    policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)
    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))
    batch_data = BatchGraph()

    # inst = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(n_generated_instances)])
    # np.save('./test_data/syn_test_instance_{}x{}.npy'.format(j, m), inst)
    # inst = np.load('./test_data/tai15x15.npy')
    inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(j, m))

    # rollout network
    print('Starting rollout DRL policy...')
    t1_drl = time.time()
    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
    while env.itr < transit:
        batch_data.wrapper(*states)
        actions, _ = policy(batch_data, feasible_actions)
        states, _, feasible_actions, _ = env.step(actions, dev)
    DRL_result = env.current_objs.cpu().squeeze().numpy()
    t2_drl = time.time()
    print('DRL results:\n', DRL_result, '\ntakes: {:.2f} per instance.'.format((t2_drl - t1_drl)/inst.shape[0]))

    # rollout random policy
    import random
    random.seed(1)
    print('Starting rollout random policy...')
    t1_random = time.time()
    states, feasible_actions, _ = env.reset(instances=inst, init_type=init, device=dev)
    while env.itr < transit:
        actions = [random.choice(feasible_action) for feasible_action in feasible_actions]
        states, _, feasible_actions, _ = env.step(actions, dev)
    Random_result = env.current_objs.cpu().squeeze().numpy()
    t2_random = time.time()
    print('Random results:\n', Random_result, '\ntakes: {:.2f} per instance.'.format((t2_random - t1_random)/inst.shape[0]))

    # ortools solver
    from pathlib import Path
    ortools_path = Path('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(j, m))
    if ortools_path.is_file():
        results_ortools = np.load('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(j, m))
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
        np.save('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(j, m), results_ortools)
    print(results_ortools)

    print('DRL Gap:', ((DRL_result - results_ortools)/results_ortools).mean())
    print('Random Gap:', ((Random_result - results_ortools) / results_ortools).mean())


if __name__ == '__main__':
    main()