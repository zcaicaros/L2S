import numpy as np
import matplotlib.pyplot as plt


# env parameters
j = 10
m = 10
l = 1
h = 99
init_type = 'fdd-divide-mwkr'
reward_type = 'yaoxin'
gamma = 1
# model parameters
hidden_dim = 128
embedding_layer = 4
policy_layer = 4
embedding_type = 'gin'
# training parameters
lr = 5e-5
steps_learn = 10
transit = 500
batch_size = 256
episodes = 128000
step_validation = 10
# plot parameters
show = True
log_type = 'training'  # 'training', 'validation'
plot_step_size_training = 1
plot_step_size_validation = 1




log = np.load('./renamed_log/{}_log_'  # log type
              '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
              '{}_{}_{}_{}_'  # model parameters
              '{}_{}_{}_{}_{}_{}.npy'  # training parameters
              .format(log_type, j, m, l, h, init_type, reward_type, gamma,
                      hidden_dim, embedding_layer, policy_layer, embedding_type,
                      lr, steps_learn, transit, batch_size, episodes,
                      step_validation))

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
    # print(obj_incumbent.min())
    # print(obj_incumbent.max())
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
    # print(obj_current.min())
    # print(obj_current.max())
    plt.xlabel('current-iteration({})'.format(plot_step_size_validation))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj_current.shape[0])], obj_current, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


