import numpy as np


plist = np.repeat(np.arange(15).repeat(15).reshape(1, -1), repeats=2, axis=0)
plist1 = np.repeat(np.random.permutation(np.arange(15).repeat(15)).reshape(1, -1), repeats=2, axis=0)
permuted_random_plist = np.random.permutation(plist)
print(plist1)

np.argmax