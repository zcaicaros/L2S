import numpy as np
import matplotlib.pyplot as plt


show = True
j = 10
m = 10
episode = 128000
transit = 128
init = 'fdd-divide-mwkr'  # 'plist', 'spt', ...
file = './log/batch_log_{}x{}_{}w_{}_{}.npy'.format(j, m, str(episode/10000), init, transit)
plot_step_size = 10
log = np.load(file)

obj = log[:log.shape[0]//plot_step_size*plot_step_size].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('make span')
plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


