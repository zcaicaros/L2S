import numpy as np
import matplotlib.pyplot as plt


show = True
plot_step_size = 100
horizon = 10000
log = np.load('log/log2.npy')


obj = log[:horizon, 0].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration')
plt.ylabel('make span')
plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


returns = log[:horizon, 1].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration')
plt.ylabel('return')
plt.plot([_ for _ in range(returns.shape[0])], returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


running_returns = log[:horizon, 2].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration')
plt.ylabel('running return')
plt.plot([_ for _ in range(running_returns.shape[0])], running_returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()