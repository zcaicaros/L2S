import numpy as np
import torch
import time
from env.env_batch import JsspN5
from model.actor import Actor
from ortools_baseline import MinimalJobshopSat
from env.env_batch import BatchGraph
import copy


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
result_type = 'incumbent'  # 'current', 'incumbent'


torch.manual_seed(1)
np.random.seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def greedy(feasible_actions, current_graph, current_tabu_list, current_obj, incumbent_obj, instance, device):

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
        best_move = [[0, 0]]

    return best_move


def main():
    env = JsspN5(n_job=p_j, n_mch=p_m, low=l, high=h, reward_type=reward_type)
    policy = Actor(3, 128, gin_l=4, policy_l=4).to(dev)
    saved_model_path = './saved_model/{}x{}_{}_{}_{}_{}_reward.pth'.format(model_j, model_m, init, training_episode_length, model_type, reward_type)
    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

    # inst = np.array([uni_instance_gen(n_j=p_j, n_m=p_m, low=l, high=h) for _ in range(n_generated_instances)])
    # np.save('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m), inst)

    if testing_type == 'tai':
        inst = np.load('./test_data/tai{}x{}.npy'.format(p_j, p_m))
    elif testing_type == 'syn':
        inst = np.load('./test_data/syn_test_instance_{}x{}.npy'.format(p_j, p_m))
    else:
        raise ValueError('testing_type must be "tai" or "syn".')

    # rollout greedy
    print('Starting rollout greedy policy...')
    t1_greedy = time.time()
    greedy_result = []
    for ins in inst[np.newaxis, np.newaxis, :, :]:
        _, feasible_actions, _ = env.reset(instances=ins, init_type=init, device=dev)

        while env.itr < transit:
            best_move = greedy(feasible_actions=feasible_actions[0],
                               current_graph=env.current_graphs[0],
                               current_tabu_list=env.tabu_lists[0],
                               current_obj=env.current_objs[0],
                               incumbent_obj=env.incumbent_objs[0],
                               instance=env.instances[0],
                               device=dev)
            _, _, feasible_actions, _ = env.step(best_move, dev)
        greedy_result.append(env.incumbent_objs.cpu().item())
    t2_greedy = time.time()
    greedy_result = np.array(greedy_result)
    print('Greedy results takes: {:.4f}s per instance.\n'.format(t2_greedy - t1_greedy), greedy_result)
    # print(env.incumbent_objs)
    # print(np.load('./test_data/ortools_result_syn_test_data_{}x{}.npy'.format(p_j, p_m)))



if __name__ == '__main__':
    main()