import pstats



p = pstats.Stats('./restats_het')
p.strip_dirs().sort_stats('cumtime').print_stats()