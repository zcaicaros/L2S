import numpy as np


problem, j, m = 'abz', 10, 10
GNNRL_result = np.array([1353, 1043], dtype=float)
ScheduleNet_result = np.array([1336, 981], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'abz', 20, 15
GNNRL_result = np.array([887, 843, 848], dtype=float)
ScheduleNet_result = np.array([791, 787, 832], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'orb', 10, 10
GNNRL_result = np.array([1336, 1067, 1202, 1281, 1082, 1178, 477, 1156, 1143, 1087], dtype=float)
ScheduleNet_result = np.array([1276, 958, 1335, 1178, 1042, 1222, 456, 1178, 1145, 1080], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'yn', 20, 20
GNNRL_result = np.array([1118, 1097, 1083, 1258], dtype=float)
ScheduleNet_result = np.array([1027, 1037, 1046, 1216], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'swv', 20, 10
GNNRL_result = np.array([1761, 1846, 1892, 1908, 1796], dtype=float)
ScheduleNet_result = np.array([1913, 1998, 1830, 1971, 1922], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'swv', 20, 15
GNNRL_result = np.array([2068, 2194, 2191, 2278, 2141], dtype=float)
ScheduleNet_result = np.array([2216, 2037, 2255, 2196, 2279], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'swv', 50, 10
GNNRL_result = np.array([3989, 4136, 4008, 3758, 3860, 2924, 2840, 2852, 2961, 2823], dtype=float)
ScheduleNet_result = np.array([4390, 4532, 4602, 4387, 4402, 2924, 2794, 2852, 2992, 2823], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 10, 5
GNNRL_result = np.array([805, 687, 862, 650, 593], dtype=float)
ScheduleNet_result = np.array([680, 768, 734, 698, 593], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 15, 5
GNNRL_result = np.array([926, 931, 863, 951, 966], dtype=float)
ScheduleNet_result = np.array([926, 1008, 863, 951, 958], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 20, 5
GNNRL_result = np.array([1276, 1039, 1150, 1292, 1282], dtype=float)
ScheduleNet_result = np.array([1254, 1039, 1150, 1292, 1395], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 10, 10
GNNRL_result = np.array([1134, 953, 1049, 880, 1042], dtype=float)
ScheduleNet_result = np.array([1047, 888, 947, 963, 989], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 15, 10
GNNRL_result = np.array([1309, 1158, 1085, 1129, 1308], dtype=float)
ScheduleNet_result = np.array([1261, 1027, 1145, 1088, 1117], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 20, 10
GNNRL_result = np.array([1553, 1624, 1438, 1582, 1649], dtype=float)
ScheduleNet_result = np.array([1458, 1516, 1357, 1320, 1490], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 30, 10
GNNRL_result = np.array([1817, 1977, 1795, 1895, 2041], dtype=float)
ScheduleNet_result = np.array([1906, 1850, 1731, 1784, 1969], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()

problem, j, m = 'la', 15, 15
GNNRL_result = np.array([1489, 1623, 1421, 1555, 1570], dtype=float)
ScheduleNet_result = np.array([1449, 1653, 1444, 1430, 1357], dtype=float)
gap_against = np.load('../test_data/{}{}x{}_result.npy'.format(problem, j, m))
gap_GNNRL = (GNNRL_result - gap_against)/gap_against
gap_ScheduleNet = (ScheduleNet_result - gap_against)/gap_against
print('GNNRL mean gap for {}{}x{}:'.format(problem, j, m), gap_GNNRL.mean())
print('ScheduleNet mean gap for {}{}x{}:'.format(problem, j, m), gap_ScheduleNet.mean())
print()






