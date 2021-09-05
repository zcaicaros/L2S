import numpy as np


# drl_result = np.load('./test_results/DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/syn_150x25_fdd-divide-mwkr_result.npy')[:, :10]
# gap_against = np.load('./test_data/syn150x25_result.npy')[:10]

# print(drl_result)
# print(gap_against)
#
# print(((drl_result - gap_against)/gap_against).mean(axis=-1))



drl_time = np.load('./test_results/DRL_results/incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/ft_20x5_fdd-divide-mwkr_time.npy')
# print(drl_time)


# problem = 'yn'
# j = 20
# m = 20
# gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(problem, j, m))
# ortools_result = np.load('./ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
# ortools_time = np.load('./ortools_result/ortools_{}{}x{}_time.npy'.format(problem, j, m))
# print(((ortools_result - gap_against)/gap_against).mean(axis=-1))
# print(ortools_time.mean())

problem = 'syn'
j = 100
m = 20
ortools_time = np.load('./test_data/{}{}x{}_time.npy'.format(problem, j, m))
print(ortools_time.mean())