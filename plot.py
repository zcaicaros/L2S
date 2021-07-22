import numpy as np
import matplotlib.pyplot as plt


show = True
j = 10
m = 10
episode = 64000
transit = 64
init = 'fdd-divide-mwkr'  # 'plist', 'spt', ...
file = './log/log_{}x{}_{}w_{}_{}.npy'.format(j, m, str(episode/10000), init, transit)
plot_step_size = 100
log = np.load(file)


obj = log[:log.shape[0]//plot_step_size*plot_step_size, 0].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot objective...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('make span')
plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


returns = log[:log.shape[0]//plot_step_size*plot_step_size, 1].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot return...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('return')
plt.plot([_ for _ in range(returns.shape[0])], returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()


'''running_returns = log[:log.shape[0]//plot_step_size*plot_step_size, 2].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
# plot running return...
plt.xlabel('iteration({})'.format(plot_step_size))
plt.ylabel('running return')
plt.plot([_ for _ in range(running_returns.shape[0])], running_returns, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()'''

validation_file = './log/validation_log_{}x{}_{}w_{}_{}.npy'.format(j, m, str(episode/10000), init, transit)
validation_log = np.load(validation_file)
# plot validation result...
plt.xlabel('iteration(per 100)')
plt.ylabel('last step make span')
plt.plot([_ for _ in range(validation_log.shape[0])], validation_log, color='tab:blue')
plt.grid()
plt.tight_layout()
if show:
    plt.show()
plt.close()