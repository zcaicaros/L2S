import pandas as pd
import numpy as np


# problem config
p_j = 10
p_m = 10
l = 1
h = 99
testing_type = 'syn'  # 'tai', 'syn'

## convert your array into a dataframe
df = pd.DataFrame(np.load('results_{}{}x{}.npy'.format(testing_type, p_j, p_m)))
## save to xlsx file
filepath = 'results_{}{}x{}.xlsx'.format(testing_type, p_j, p_m)
df.to_excel(filepath, index=False)

## convert your array into a dataframe
df = pd.DataFrame(np.load('inference_time_{}{}x{}.npy'.format(testing_type, p_j, p_m)))
## save to xlsx file
filepath = 'inference_time_{}{}x{}.xlsx'.format(testing_type, p_j, p_m)
df.to_excel(filepath, index=False)
