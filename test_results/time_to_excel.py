import pandas as pd
import numpy as np


baseline = ['incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'incumbent_15x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'incumbent_15x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'greedy', 'best-improvement', 'first-improvement']
init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
testing_type = ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn']  # ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn']
syn_problem_j = [10, 15, 15, 20, 20, 100, 150]  # [10, 15, 15, 20, 20, 100, 200]
syn_problem_m = [10, 10, 15, 10, 15, 20, 25]  # [10, 10, 15, 10, 15, 20, 50]
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

for method in baseline:
    mean_time_all_dataset = []
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
            raise Exception('Problem type must be in testing_type = ["tai", "abz", "orb", "yn", "swv", "la", "ft", "syn"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size
            for init in init_type:
                if method in ['greedy', 'best-improvement', 'first-improvement']:
                    baseline_time = np.load('./conventional_results/{}-policy/{}{}x{}_{}_time.npy'.format(method, test_t, p_j, p_m, init))
                    mean_time_all_dataset.append(baseline_time.reshape(-1, 1))
                else:
                    baseline_time = np.load(
                        './DRL_results/{}/{}_{}x{}_{}_time.npy'.format(method, test_t, p_j, p_m, init))
                    mean_time_all_dataset.append(baseline_time.reshape(-1, 1))
        mean_time_all_dataset.append(-np.ones(shape=[1, len(init_type)], dtype=float))

    # print(mean_time_all_dataset)
    time_to_excel = np.concatenate(mean_time_all_dataset, axis=0)
    df = pd.DataFrame(time_to_excel)
    df.to_excel('{}_time.xlsx'.format(method), index=False)
