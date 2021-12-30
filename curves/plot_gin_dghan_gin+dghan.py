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
embedding_type1 = 'gin'  # 'gin', 'dghan', 'gin+dghan'
embedding_type2 = 'dghan'  # 'gin', 'dghan', 'gin+dghan'
embedding_type3 = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
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
show = True
save = True
log_type = 'training'  # 'training', 'validation'
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'


if embedding_type1 == 'gin':
    dghan_param_for_saved_model1 = 'NAN'
elif embedding_type1 == 'dghan' or embedding_type1 == 'gin+dghan':
    dghan_param_for_saved_model1 = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

if embedding_type2 == 'gin':
    dghan_param_for_saved_model2 = 'NAN'
elif embedding_type2 == 'dghan' or embedding_type2 == 'gin+dghan':
    dghan_param_for_saved_model2 = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

if embedding_type3 == 'gin':
    dghan_param_for_saved_model3 = 'NAN'
elif embedding_type3 == 'dghan' or embedding_type3 == 'gin+dghan':
    dghan_param_for_saved_model3 = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

file1 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type1, dghan_param_for_saved_model1,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log1 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file1 + '.npy')

file2 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type2, dghan_param_for_saved_model2,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log2 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file2 + '.npy')

file3 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type3, dghan_param_for_saved_model3,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log3 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file3 + '.npy')

if log_type == 'training':
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj2 = log2[:log2.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log2.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj3 = log3[:log3.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log3.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_training), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Makespan', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj1.shape[0])])
    plt.plot(x, obj1, color='tab:blue', label='TPM: {}×{}'.format(j, m))
    plt.plot(x, obj2, color='tab:red', label='CAM: {}×{}'.format(j, m))
    plt.plot(x, obj3, color='tab:green', label='TPM + CAM: {}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_training_log', save_file_type))
    if show:
        plt.show()

else:
    obj_incumbent1 = log1[:log1.shape[0]//plot_step_size_validation*plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent3 = log3[:log3.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log3.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Makespan', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x, obj_incumbent1, color='tab:blue', label='TPM: {}×{}'.format(j, m))
    plt.plot(x, obj_incumbent2, color='tab:red', label='CAM: {}×{}'.format(j, m))
    plt.plot(x, obj_incumbent3, color='tab:green', label='TPM + CAM: {}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_incumbent_validation_log', save_file_type))
    if show:
        plt.show()

    obj_last_step1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step3 = log3[:log3.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log3.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Makespan', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_last_step1.shape[0])])
    plt.plot(x, obj_last_step1, color='tab:blue', label='TPM: {}×{}'.format(j, m))
    plt.plot(x, obj_last_step2, color='tab:red', label='CAM: {}×{}'.format(j, m))
    plt.plot(x, obj_last_step3, color='tab:green', label='TPM + CAM: {}×{}'.format(j, m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_last-step_validation_log', save_file_type))
    if show:
        plt.show()



