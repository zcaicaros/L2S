import numpy as np
import matplotlib.pyplot as plt


show = True
init = 'fdd-divide-mwkr'  # 'plist', 'spt', ...
j = 10
m = 10
episode = 128000
training_episode_length = 256
reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
log_type = 'validation'  # 'validation', 'training'
plot_step_size_training = 10
plot_step_size_validation = 2


file = './log/batch_{}_log_{}x{}_{}w_{}_{}_{}_reward.npy'.format(log_type, j, m, str(episode/10000), init, training_episode_length, reward_type)
log = np.load(file)

if log_type == 'training':
    obj = log[:log.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plot objective...
    plt.xlabel('iteration({})'.format(plot_step_size_training))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

else:
    obj_incumbent = log[:log.shape[0]//plot_step_size_validation*plot_step_size_validation, 0].reshape(log.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plot objective...
    plt.xlabel('incumbent-iteration({})'.format(plot_step_size_validation))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj_incumbent.shape[0])], obj_incumbent, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

    obj_current = log[:log.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plot objective...
    plt.xlabel('current-iteration({})'.format(plot_step_size_validation))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj_current.shape[0])], obj_current, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


