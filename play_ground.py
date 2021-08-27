import numpy as np


print(np.load('./ortools_result/ortools_{}{}x{}_result.npy'.format('tai', 30, 20)))
print(np.load('./ortools_result/ortools_{}{}x{}_time.npy'.format('tai', 30, 20)))

print(np.load('./test_results/DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_4e-05_10_500_64_256000_10/{}_{}x{}_{}_result.npy'.format('abz', 10, 10, 'fdd-divide-mwkr')))
print(np.load('./test_results/DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_4e-05_10_500_64_256000_10/{}_{}x{}_{}_time.npy'.format('abz', 10, 10, 'fdd-divide-mwkr')))


np.save('./test_data/{}{}x{}_time.npy'.format('syn', 100, 20), 3600*np.ones(100).reshape(-1, 1))