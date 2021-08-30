import numpy as np


# print(np.load('./ortools_result/ortools_{}{}x{}_result.npy'.format('tai', 30, 20)))
# print(np.load('./ortools_result/ortools_{}{}x{}_time.npy'.format('tai', 30, 20)))
#
# print(np.load('./test_results/DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_4e-05_10_500_64_256000_10/{}_{}x{}_{}_result.npy'.format('abz', 10, 10, 'fdd-divide-mwkr')))
# print(np.load('./test_results/DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_4e-05_10_500_64_256000_10/{}_{}x{}_{}_time.npy'.format('abz', 10, 10, 'fdd-divide-mwkr')))

print(np.load('./test_results/conventional_results/swv20x15_best-improvement-policy_fdd-divide-mwkr_result.npy'))

a = [1, 2]
import random
a_ele = random.choice(a)
print(a_ele)
a_ele = 3
print(a)
a_in = a[::-1]
print(a)