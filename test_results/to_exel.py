import pandas as pd
import numpy as np


# problem config
baseline = 'greedy'  # 'greedy', 'best-improvement', or 'first-improvement'
init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
testing_type = ['syn', 'tai', 'abz']  # ['syn', 'tai', 'abz', 'orb', 'yn', 'swv', 'la']
syn_problem_j = [15]  # [10, 15, 20, 30, 50, 100]
syn_problem_m = [15]  # [10, 15, 20, 20, 20, 20]
tai_problem_j = [15, 20]  # [15, 20, 20, 30, 30, 50, 50, 100]
tai_problem_m = [15, 15]  # [15, 15, 20, 15, 20, 15, 20, 20]
abz_problem_j = [10, 20]  # [10, 20]
abz_problem_m = [10, 15]  # [10, 15]
orb_problem_j = [10]  # [10]
orb_problem_m = [10]  # [10]
yn_problem_j = [20]  # [20]
yn_problem_m = [20]  # [20]
swv_problem_j = [20, 20, 50]  # [20, 20, 50]
swv_problem_m = [10, 15, 10]  # [10, 15, 10]
la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]

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
    else:
        raise Exception('Problem type must be in testing_type = ["syn", "tai", "abz", "orb", "yn", "swv", "la"].')

    for p_j, p_m in zip(problem_j, problem_m):  # select problem size
        for init in init_type:
            print(np.load('./conventional_results/{}{}x{}_greedy-policy_{}_results.npy'.format(test_t, p_j, p_m, init)))