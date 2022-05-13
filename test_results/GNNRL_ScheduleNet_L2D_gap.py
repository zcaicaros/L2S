import numpy as np


problem, j, m = 'tai', 15, 15
GNNRL_result = np.array([1389, 1519, 1457, 1465, 1352, 1481, 1554, 1488, 1556, 1501], dtype=float)
ScheduleNet_result = np.array([1452, 1411, 1396, 1348, 1382, 1413, 1380, 1374, 1523, 1493], dtype=float)
L2D_result = np.array([1443, 1544, 1440, 1637, 1619, 1601, 1568, 1468, 1627, 1527], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_15x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 20, 15
GNNRL_result = np.array([1626, 1668, 1715, 1642, 1672, 1700, 1678, 1684, 1900, 1752], dtype=float)
ScheduleNet_result = np.array([1612, 1600, 1625, 1590, 1676, 1550, 1753, 1668, 1622, 1604], dtype=float)
L2D_result = np.array([1794, 1805, 1932, 1664, 1730, 1710, 1897, 1794, 1682, 1739], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 20, 20
GNNRL_result = np.array([2199, 2049, 2006, 2020, 1981, 2057, 2187, 2054, 2210, 2140], dtype=float)
ScheduleNet_result = np.array([1921, 1844, 1879, 1922, 1897, 1887, 2009, 1813, 1875, 1913], dtype=float)
L2D_result = np.array([2252, 2102, 2085, 2200, 2201, 2176, 2132, 2146, 1952, 2035], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 30, 15
GNNRL_result = np.array([2251, 2378, 2316, 2319, 2333, 2210, 2201, 2151, 2138, 2007], dtype=float)
ScheduleNet_result = np.array([2055, 2268, 2281, 2061, 2218, 2154, 2112, 1970, 2146, 2030], dtype=float)
L2D_result = np.array([2565, 2388, 2324, 2332, 2505, 2497, 2325, 2302, 2410, 2140], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 30, 20
GNNRL_result = np.array([2654, 2579, 2737, 2772, 2435, 2681, 2428, 2440, 2446, 2530], dtype=float)
ScheduleNet_result = np.array([2572, 2397, 2310, 2456, 2445, 2541, 2280, 2358, 2301, 2453], dtype=float)
L2D_result = np.array([2667, 2664, 2431, 2714, 2637, 2776, 2476, 2490, 2556, 2628], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 50, 15
GNNRL_result = np.array([3145, 3157, 3103, 3278, 3142, 3258, 3230, 3469, 3108, 3256], dtype=float)
ScheduleNet_result = np.array([3382, 3231, 3083, 3068, 3078, 3065, 3266, 3321, 3044, 3036], dtype=float)
L2D_result = np.array([3599, 3341, 3106, 3266, 3232, 3378, 3471, 3454, 3381, 3281], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 50, 20
GNNRL_result = np.array([3425, 3626, 3110, 3329, 3339, 3340, 3371, 3265, 3798, 3919], dtype=float)
ScheduleNet_result = np.array([3202, 3339, 3118, 2989, 3168, 3199, 3236, 3072, 3535, 3436], dtype=float)
L2D_result = np.array([3654, 3617, 3397, 3275, 3359, 3388, 3567, 3514, 3592, 3643], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'tai', 100, 20
GNNRL_result = np.array([5962, 5522, 6335, 5827, 6042, 5707, 5737, 5979, 5799, 5718], dtype=float)
ScheduleNet_result = np.array([5879, 5456, 6052, 5513, 5992, 5773, 5637, 5833, 5556, 5545], dtype=float)
L2D_result = np.array([6452, 5695, 6411, 5885, 6355, 6135, 6056, 6101, 5943, 5892], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'abz', 10, 10
GNNRL_result = np.array([1353, 1043], dtype=float)
ScheduleNet_result = np.array([1336, 981], dtype=float)
L2D_result = np.array([1401, 1162], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'abz', 20, 15
GNNRL_result = np.array([887, 843, 848], dtype=float)
ScheduleNet_result = np.array([791, 787, 832], dtype=float)
L2D_result = np.array([815, 858, 902], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()


problem, j, m = 'ft', 6, 6
GNNRL_result = np.array([71], dtype=float)
ScheduleNet_result = np.array([59], dtype=float)
L2D_result = np.array([66], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_15x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'ft', 10, 10
GNNRL_result = np.array([1142], dtype=float)
ScheduleNet_result = np.array([1111], dtype=float)
L2D_result = np.array([1271], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'ft', 20, 5
GNNRL_result = np.array([1338], dtype=float)
ScheduleNet_result = np.array([1498], dtype=float)
L2D_result = np.array([1614], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 10, 5
GNNRL_result = np.array([805, 687, 862, 650, 593], dtype=float)
ScheduleNet_result = np.array([680, 768, 734, 698, 593], dtype=float)
L2D_result = np.array([749, 812, 811, 827, 593], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/(gap_L2D + 1e-20)).mean()))  # L2D get one optimal solution
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 15, 5
GNNRL_result = np.array([926, 931, 863, 951, 966], dtype=float)
ScheduleNet_result = np.array([926, 1008, 863, 951, 958], dtype=float)
L2D_result = np.array([1021, 1001, 991, 951, 958], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_15x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/(gap_L2D + 1e-20)).mean()))  # L2D get one optimal solution for last 2 instance
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 20, 5
GNNRL_result = np.array([1276, 1039, 1150, 1292, 1282], dtype=float)
ScheduleNet_result = np.array([1254, 1039, 1150, 1292, 1395], dtype=float)
L2D_result = np.array([1225, 1089, 1182, 1292, 1486], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/(gap_L2D + 1e-20)).mean()))  # L2D get one optimal solution for forth instance
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 10, 10
GNNRL_result = np.array([1134, 953, 1049, 880, 1042], dtype=float)
ScheduleNet_result = np.array([1047, 888, 947, 963, 989], dtype=float)
L2D_result = np.array([1193, 935, 1060, 1018, 1126], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 15, 10
GNNRL_result = np.array([1309, 1158, 1085, 1129, 1308], dtype=float)
ScheduleNet_result = np.array([1261, 1027, 1145, 1088, 1117], dtype=float)
L2D_result = np.array([1237, 1207, 1218, 1241, 1211], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_15x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 20, 10
GNNRL_result = np.array([1553, 1624, 1438, 1582, 1649], dtype=float)
ScheduleNet_result = np.array([1458, 1516, 1357, 1320, 1490], dtype=float)
L2D_result = np.array([1452, 1584, 1603, 1534, 1675], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 30, 10
GNNRL_result = np.array([1817, 1977, 1795, 1895, 2041], dtype=float)
ScheduleNet_result = np.array([1906, 1850, 1731, 1784, 1969], dtype=float)
L2D_result = np.array([1980, 2084, 1850, 1935, 2142], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'la', 15, 15
GNNRL_result = np.array([1489, 1623, 1421, 1555, 1570], dtype=float)
ScheduleNet_result = np.array([1449, 1653, 1444, 1430, 1357], dtype=float)
L2D_result = np.array([1752, 1660, 1619, 1628, 1644], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_15x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'swv', 20, 10
GNNRL_result = np.array([1761, 1846, 1892, 1908, 1796], dtype=float)
ScheduleNet_result = np.array([1913, 1998, 1830, 1971, 1922], dtype=float)
L2D_result = np.array([1984, 2208, 2083, 2256, 2097], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'swv', 20, 15
GNNRL_result = np.array([2068, 2194, 2191, 2278, 2141], dtype=float)
ScheduleNet_result = np.array([2216, 2037, 2255, 2196, 2279], dtype=float)
L2D_result = np.array([2554, 2289, 2422, 2413, 2423], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'swv', 50, 10
GNNRL_result = np.array([3989, 4136, 4008, 3758, 3860, 2924, 2840, 2852, 2961, 2823], dtype=float)
ScheduleNet_result = np.array([4390, 4532, 4602, 4387, 4402, 2924, 2794, 2852, 2992, 2823], dtype=float)
L2D_result = np.array([4503, 4708, 4827, 4801, 4760, 3058, 2853, 2937, 3166, 2858], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'orb', 10, 10
GNNRL_result = np.array([1336, 1067, 1202, 1281, 1082, 1178, 477, 1156, 1143, 1087], dtype=float)
ScheduleNet_result = np.array([1276, 958, 1335, 1178, 1042, 1222, 456, 1178, 1145, 1080], dtype=float)
L2D_result = np.array([1366, 1052, 1443, 1308, 1170, 1372, 491, 1291, 1111, 1365], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()

problem, j, m = 'yn', 20, 20
GNNRL_result = np.array([1118, 1097, 1083, 1258], dtype=float)
ScheduleNet_result = np.array([1027, 1037, 1046, 1216], dtype=float)
L2D_result = np.array([1161, 1193, 1101, 1248], dtype=float)
L2S_result = np.load('./DRL_results/incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10/{}_{}x{}_fdd-divide-mwkr_result.npy'.format(problem, j, m))[0]
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
gap_L2D = (L2D_result - gap_against)/gap_against
gap_L2S = (L2S_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print('L2D mean gap for {}{}x{}:'.format(problem, j, m), gap_L2D.mean())
print('L2S mean gap for {}{}x{}:'.format(problem, j, m), gap_L2S.mean())
print('L2S is better than L2D by {} in terms of optimality gap.'.format(((gap_L2D - gap_L2S)/gap_L2D).mean()))
Ortools_result = np.load('../ortools_result/ortools_{}{}x{}_result.npy'.format(problem, j, m))[:, 1]
gap_ortools = (Ortools_result - gap_against)/gap_against
print('Ortools mean gap for {}{}x{}:'.format(problem, j, m), gap_ortools.mean())
print()





