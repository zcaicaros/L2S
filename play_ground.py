import numpy as np


# print(np.load('./log/training_log_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10.npy')[:40])

a = np.load('test_results/DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/tai_30x20_fdd-divide-mwkr_result.npy')[0]
b = np.load('test_data/tai20x15_result.npy')
onlineDRL = np.array([2208, 2168, 2086, 2261, 2227, 2349, 2101, 2267, 2154, 2216], dtype=np.float32)

print(((onlineDRL - b)/b).mean())
print(((a - b)/b).mean())
print(a)

print(np.random.randint(1, 101))