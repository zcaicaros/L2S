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
embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
heads = 1
drop_out = 0

# training parameters
lr = 5e-5
steps_learn = 10
transit = 500
batch_size = 64
episodes = 128000
step_validation = 10

# plot parameters
show = True
save = False
log_type = 'validation'  # 'training', 'validation'
plot_step_size_training = 1
plot_step_size_validation = 1


if embedding_type == 'gin':
    dghan_param_for_saved_model = 'NAN'
elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
    dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

file = '{}x{}[{},{}]_{}_{}_{}_' \
       '{}_{}_{}_{}_{}_' \
       '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log = np.load('./log/OLD_'
              '{}_log_'
              .format(log_type)  # log type
              + file + '.npy')

if log_type == 'training':
    obj = log[:log.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plot objective...
    # print(obj.min())
    # print(obj[0:100])
    # print(obj)
    plt.xlabel('iteration(stride-{})'.format(plot_step_size_training))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj.shape[0])], obj, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig('./curves/{}_plt_'
                    .format(log_type)  # log type
                    + file + '.png')
    if show:
        plt.show()
    plt.close()

else:
    obj_incumbent = log[:log.shape[0]//plot_step_size_validation*plot_step_size_validation, 0].reshape(log.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plot objective...
    print(obj_incumbent)
    # print(obj_incumbent.min())
    # print(obj_incumbent.max())
    plt.xlabel('incumbent-iteration(stride-{})'.format(plot_step_size_validation))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj_incumbent.shape[0])], obj_incumbent, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig('./curves/{}_plt_incumbent_'
                    .format(log_type)  # log type
                    + file + '.png')
    if show:
        plt.show()
    plt.close()

    obj_last_step = log[:log.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plot objective...
    # print(obj_last_step)
    # print(obj_last_step.min())
    # print(obj_last_step.max())
    plt.xlabel('last-step-iteration(stride-{})'.format(plot_step_size_validation))
    plt.ylabel('make span')
    plt.plot([_ for _ in range(obj_last_step.shape[0])], obj_last_step, color='tab:blue')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig('./curves/{}_plt_last-step_'
                    .format(log_type)  # log type
                    + file + '.png')
    if show:
        plt.show()
    plt.close()


