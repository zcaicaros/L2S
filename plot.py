import numpy as np
import matplotlib.pyplot as plt


show = True
j = 10
m = 10
init = 'spt'  # 'plist', 'spt', ...
episode = '38.4w'  # '25.6w', '38.4w', ...
file = 'log/log_{}x{}_sample_{}_{}.npy'.format(j, m, episode, init)
plot_step_size = 1000
horizon = 127000
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