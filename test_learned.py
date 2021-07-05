import numpy as np
import torch
from env.env_single import JsspN5
from model.actor import Actor
from env.generateJSP import uni_instance_gen
from torch_geometric.data.batch import Batch
from ortools_baseline import MinimalJobshopSat


show = True
j = 10
m = 10
l = 1
h = 99
n_generated_instances = 100
transit = 2000
init = 'rule'  # 'plist', 'spt', ...
rule = 'spt'

torch.manual_seed(1)
np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    if init == 'p_list':
        saved_model_path = './saved_model/{}x{}_plist.pth'.format(j, m)
    else:
        saved_model_path = './saved_model/{}x{}_{}.pth'.format(j, m, rule)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, transition=transit, init=init, rule=rule)
    policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)
    # policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

    # inst = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(n_generated_instances)])
    # np.save('./test_data/syn_test_instance_{}x{}.npy'.format(j, m), inst)
    # inst = np.load('./test_data/tai15x15.npy')
    inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(j, m))

    # rollout network
    results_drl = []
    for i, data in enumerate(inst):
        state, feasible_action, done = env.reset(instance=data, fix_instance=True)
        returns = []
        with torch.no_grad():
            while not done:
                action, _ = policy(Batch.from_data_list([state]).to(dev), [feasible_action])
                state, reward, feasible_action, done = env.step_single(action=action[0])
                returns.append(reward)
        print('Instance-' + str(i + 1) + ' DRL makespan:', env.incumbent_obj, ' used transition:', env.itr)
        results_drl.append(env.incumbent_obj)
    results_drl = np.array(results_drl)

    # rollout random policy
    import random
    random.seed(1)
    results_random = []
    for i, data in enumerate(inst):
        state, feasible_action, done = env.reset(instance=data, fix_instance=True)
        returns = []
        with torch.no_grad():
            while not done:
                action = random.choice(feasible_action)
                state, reward, feasible_action, done = env.step_single(action=action)
                returns.append(reward)
        print('Instance-' + str(i + 1) + ' Random policy makespan:', env.incumbent_obj, ' used transition:', env.itr)
        results_random.append(env.incumbent_obj)
    results_random = np.array(results_random)

    # ortools solver
    results_ortools = []
    for i, data in enumerate(inst):
        times_rearrange = np.expand_dims(data[0], axis=-1)
        machines_rearrange = np.expand_dims(data[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        result = MinimalJobshopSat(data.tolist())
        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
        results_ortools.append(result[1])
    results_ortools = np.array(results_ortools)

    print('DRL Gap:', ((results_drl - results_ortools)/results_ortools).mean())
    print('Random Gap:', ((results_random - results_ortools) / results_ortools).mean())


if __name__ == '__main__':
    main()