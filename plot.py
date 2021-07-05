import numpy as np
import matplotlib.pyplot as plt
from parameters import args


show = True
j = 15
m = 15
episode = 384000
init = 'spt'  # 'plist', 'spt', ...
file = 'log/log_{}x{}_sample_{}_{}.npy'.format(j, m, str(episode/10000) + 'w', init)
plot_step_size = 200
horizon = None
if horizon is not None:
    log = np.load(file)[:horizon]
else:
    log = np.load(file)


obj = log[:, 0].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('make span')
plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


returns = log[:, 1].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot return...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('return')
plt.plot([_ for _ in range(returns.shape[0])], returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


'''running_returns = log[:, 2].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot running return...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('running return')
plt.plot([_ for _ in range(running_returns.shape[0])], running_returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()'''