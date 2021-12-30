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
dghan_param_for_saved_model1 = '1_0.0'
dghan_param_for_saved_model2 = '2_0.0'
heads = 1
drop_out = 0.

# training parameters
lr = 5e-5
steps_learn = 10
transit = 500
batch_size = 64
episodes = 128000
step_validation = 10

# plot parameters
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17
total_plt_steps = 200
show = False
save = True
log_type = 'training'  # 'training', 'validation'
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'




file1 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model1,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log1 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file1 + '.npy')

file2 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model2,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log2 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file2 + '.npy')


if log_type == 'training':
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj2 = log2[:log2.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log2.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_training), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Makespan', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj1.shape[0])])
    plt.plot(x, obj1, color='tab:blue', label='TPM + CAM-1-Head: {}x{}'.format(j, m))
    plt.plot(x, obj2, color='tab:red', label='TPM + CAM-2-Head: {}x{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('compare_heads_training', save_file_type))
    if show:
        plt.show()

else:
    obj_incumbent1 = log1[:log1.shape[0]//plot_step_size_validation*plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('make span', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x, obj_incumbent1, color='tab:blue', label='TPM + CAM-1-Head: {}x{}'.format(j, m))
    plt.plot(x, obj_incumbent2, color='tab:red', label='TPM + CAM-2-Head: {}x{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('compare_heads_validation_incumbent', save_file_type))
    if show:
        plt.show()

    obj_last_step1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('make span', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_last_step1.shape[0])])
    plt.plot(x, obj_last_step1, color='tab:blue', label='TPM + CAM-1-Head: {}x{}'.format(j, m))
    plt.plot(x, obj_last_step2, color='tab:red', label='TPM + CAM-2-Head: {}x{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('compare_heads_validation_last-step', save_file_type))
    if show:
        plt.show()



