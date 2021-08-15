import numpy as np


plist = np.repeat(np.arange(15).repeat(15).reshape(1, -1), repeats=2, axis=0)
plist1 = np.repeat(np.random.permutation(np.arange(15).repeat(15)).reshape(1, -1), repeats=2, axis=0)
permuted_random_plist = np.random.permutation(plist)
# print(plist1)

# print(np.load('test_data/abz10x10.npy').shape)
# print(np.load('test_data/syn10x10.npy').shape)

j = 20
m = 20
dataset_type = 'yn'
data_set = np.load('test_data/{}{}x{}.npy'.format(dataset_type, j, m)).reshape(-1, 2, j, m)
print(data_set.shape)
np.save('test_data/{}{}x{}.npy'.format(dataset_type, j, m), data_set)