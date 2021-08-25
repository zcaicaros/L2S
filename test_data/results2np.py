import numpy as np

problem = 'ft'
j = 20
m = 5
upper_bounds = np.array([1165], dtype=float)
print(upper_bounds)
np.save('{}{}x{}_result.npy'.format(problem, j, m), upper_bounds)