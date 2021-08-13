import numpy as np

problem = 'yn'
j = 20
m = 20
results = np.array([884, 904, 892, 968], dtype=float)
print(results)
np.save('{}{}x{}_result.npy'.format(problem, j, m), results)