import numpy as np

problem = 'yn'
j = 20
m = 20
upper_bounds = np.array([884, 904, 892, 968], dtype=float)
print(upper_bounds)
np.save('{}{}x{}_result.npy'.format(problem, j, m), upper_bounds)