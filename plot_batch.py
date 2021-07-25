import numpy as np
import matplotlib.pyplot as plt


show = True
init = 'fdd-divide-mwkr'  # 'plist', 'spt', ...
j = 10
m = 10
episode = 128000
training_episode_length = 64
reward_type = 'consecutive'  # 'yaoxin', 'consecutive'
log_type = 'training'  # 'validation', 'training'
plot_step_size = 10


file = './log/batch_{}_log_{}x{}_{}w_{}_{}_{}_reward.npy'.format(log_type, j, m, str(episode/10000), init, training_episode_length, reward_type)
log = np.load(file)

if log_type == 'training':
    obj = log[:log.shape[0] // plot_step_size * plot_step_size].reshape(log.shape[0] // plot_step_size, -1).mean(axis=1)
else:
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
