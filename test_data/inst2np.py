import pandas
import numpy as np


insts = pandas.read_excel('./instance2numpy.xlsx', header=None).to_numpy()
dataset = 'la'
j = 15
m = 15
insts = insts.reshape(-1, 2, j, m)
np.save('{}{}x{}.npy'.format(dataset, j, m), insts)
print(np.load('{}{}x{}.npy'.format(dataset, j, m))[0])