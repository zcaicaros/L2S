import numpy as np


print(np.load('./ortools_result/ortools_{}{}x{}_result.npy'.format('tai', 30, 20)).shape)
print(np.load('./ortools_result/ortools_{}{}x{}_time.npy'.format('tai', 30, 20)).shape)
np.save('./test_data/{}{}x{}_time.npy'.format('syn', 100, 20), 3600*np.ones(100).reshape(-1, 1))