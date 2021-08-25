import pandas
import numpy as np


insts = pandas.read_excel('./instance2numpy.xlsx', header=None).to_numpy()
dataset = 'ft'
j = 20
m = 5
insts = insts.reshape(-1, 2, j, m)
np.save('{}{}x{}.npy'.format(dataset, j, m), insts)
print(np.load('{}{}x{}.npy'.format(dataset, j, m))[0])