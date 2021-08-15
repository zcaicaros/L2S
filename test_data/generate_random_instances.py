import numpy as np
from env.generateJSP import uni_instance_gen


j = 20
m = 15
l = 1
h = 99
size = 100
instances = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(size)])
np.save('syn{}x{}.npy'.format(j, m), instances)