import numpy as np
import pandas

JSSPENV = np.array([2208, 2168, 2086, 2261, 2227, 2349, 2101, 2267, 2154, 2216])
gap_against = np.load('./test_data/{}{}x{}_result.npy'.format('tai', 30, 20))
JSSPENV_gap = (JSSPENV - gap_against) / gap_against
# print(JSSPENV_gap.mean())

df = pandas.read_excel('./instance2numpy.xlsx')
print(df)