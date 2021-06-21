import pstats

n_j = 100
n_m = 20
l = 1
h = 99
max_itr = 100


p = pstats.Stats('./restats_{}x{}_{}'.format(str(n_j), str(n_m), str(max_itr)))
p.strip_dirs().sort_stats('cumtime').print_stats()