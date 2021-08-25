import numpy as np


print(np.load('./test_data/{}{}x{}.npy'.format('syn', 200, 50)).shape)
np.save('./test_data/{}{}x{}_time.npy'.format('syn', 100, 20), 3600*np.ones(100).reshape(-1, 1))