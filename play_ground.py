import numpy as np

a1 = np.array([[5, 2], [3, 4]])
a2 = np.array([[2, 2], [2, 2]])

res = (a1 - a2) / a2
print(res)
print(res.mean(axis=1))

l1 = [1]
l2 = l1 + [0]
print(l1)
print(l2)

print(np.load('test_data/syn20x10.npy').shape)