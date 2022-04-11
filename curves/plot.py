import numpy as np
import matplotlib.pyplot as plt


j = 20  # 10， 15， 15， 20， 20
m = 15  # 10， 10， 15， 10， 15
l = 1
h = 99
init_type = 'fdd-divide-mwkr'
reward_type = 'yaoxin'  # 'yaoxin', 'consecutive'
gamma = 1

hidden_dim = 128
embedding_layer = 4
policy_layer = 4
embedding_type = 'gin+dghan'  # 'gin', 'dghan', 'gin+dghan'
heads = 1
drop_out = 0.

lr = 5e-5  # 5e-5, 4e-5
steps_learn = 10
training_episode_length = 500
batch_size = 64
episodes = 128000  # 128000, 256000
step_validation = 10

# plot parameters
total_plt_steps = 200
show = True
save = False
log_type = 'validation'  # 'training', 'validation'
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'
x_label_scale = 17
y_label_scale = 17
anchor_text_size = 17


if embedding_type == 'gin':
    dghan_param_for_saved_model = 'NAN'
elif embedding_type == 'dghan' or embedding_type == 'gin+dghan':
    dghan_param_for_saved_model = '{}_{}'.format(heads, drop_out)
else:
    raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

# 10x10
file1 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(10, 10, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
log1 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file1 + '.npy')
# log1 = (log1 - np.load('../test_data/syn10x10_result.npy').mean())/np.load('../test_data/syn10x10_result.npy').mean()
# log1 = (log1 - np.load('../validation_data/validation10x10_ortools_result.npy').mean())/np.load('../test_data/syn10x10_result.npy').mean()
log1 = (log1 - np.load('../validation_data/validation10x10_ortools_result.npy').mean())/np.load('../validation_data/validation10x10_ortools_result.npy').mean()

# 15x10
file2 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(15, 10, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
log2 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file2 + '.npy')
# log2 = (log2 - np.load('../test_data/syn15x10_result.npy').mean())/np.load('../test_data/syn15x10_result.npy').mean()
# log2 = (log2 - np.load('../validation_data/validation15x10_ortools_result.npy').mean())/np.load('../test_data/syn15x10_result.npy').mean()
log2 = (log2 - np.load('../validation_data/validation15x10_ortools_result.npy').mean())/np.load('../validation_data/validation15x10_ortools_result.npy').mean()

# 15x15
file3 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(15, 15, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
log3 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file3 + '.npy')
# log3 = (log3 - np.load('../test_data/syn15x15_result.npy').mean())/np.load('../test_data/syn15x15_result.npy').mean()
# log3 = (log3 - np.load('../validation_data/validation15x15_ortools_result.npy').mean())/np.load('../test_data/syn15x15_result.npy').mean()
log3 = (log3 - np.load('../validation_data/validation15x15_ortools_result.npy').mean())/np.load('../validation_data/validation15x15_ortools_result.npy').mean()

# 20x10
file4 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(20, 10, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
log4 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file4 + '.npy')
# log4 = (log4 - np.load('../test_data/syn20x10_result.npy').mean())/np.load('../test_data/syn20x10_result.npy').mean()
# log4 = (log4 - np.load('../validation_data/validation20x10_ortools_result.npy').mean())/np.load('../test_data/syn20x10_result.npy').mean()
log4 = (log4 - np.load('../validation_data/validation20x10_ortools_result.npy').mean())/np.load('../validation_data/validation20x10_ortools_result.npy').mean()

# 20x15
file5 = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(20, 15, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, dghan_param_for_saved_model,
            lr, steps_learn, training_episode_length, batch_size, episodes, step_validation)
log5 = np.load('../log/'
               '{}_log_'
               .format(log_type)  # log type
               + file5 + '.npy')
# log5 = (log5 - np.load('../test_data/syn20x15_result.npy').mean())/np.load('../test_data/syn20x15_result.npy').mean()
# log5 = (log5 - np.load('../validation_data/validation20x15_ortools_result.npy').mean())/np.load('../test_data/syn20x15_result.npy').mean()
log5 = (log5 - np.load('../validation_data/validation20x15_ortools_result.npy').mean())/np.load('../validation_data/validation20x15_ortools_result.npy').mean()


if log_type == 'training':
    obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj2 = log2[:log2.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log2.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj3 = log3[:log3.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log3.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj4 = log4[:log4.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log4.shape[0] // plot_step_size_training, -1).mean(axis=1)
    obj5 = log5[:log5.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log5.shape[0] // plot_step_size_training, -1).mean(axis=1)
    # plotting...
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj1.shape[0])])
    plt.plot(x, obj1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.plot(x, obj2, color='tab:red', label='{}×{}'.format(15, 10))
    plt.plot(x, obj3, color='tab:green', label='{}×{}'.format(15, 15))
    plt.plot(x, obj4, color='tab:brown', label='{}×{}'.format(20, 10))
    plt.plot(x, obj5, color='tab:orange', label='{}×{}'.format(20, 15))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_training_log', save_file_type))
    if show:
        plt.show()

else:
    obj_incumbent1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent3 = log3[:log3.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log3.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent4 = log4[:log4.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log4.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_incumbent5 = log5[:log5.shape[0] // plot_step_size_validation * plot_step_size_validation, 0].reshape(log5.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_incumbent1.shape[0])])
    plt.plot(x, obj_incumbent1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.plot(x, obj_incumbent2, color='tab:red', label='{}×{}'.format(15, 10))
    plt.plot(x, obj_incumbent3, color='tab:green', label='{}×{}'.format(15, 15))
    plt.plot(x, obj_incumbent4, color='tab:brown', label='{}×{}'.format(20, 10))
    plt.plot(x, obj_incumbent5, color='tab:orange', label='{}×{}'.format(20, 15))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_incumbent_validation_log', save_file_type))
    if show:
        plt.show()

    obj_last_step1 = log1[:log1.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log1.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step3 = log3[:log3.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log3.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step4 = log4[:log4.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log4.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    obj_last_step5 = log5[:log5.shape[0] // plot_step_size_validation * plot_step_size_validation, 1].reshape(log5.shape[0] // plot_step_size_validation, -1).mean(axis=1)
    # plotting...
    # plt.xlabel('Iteration(stride-{})'.format(plot_step_size_validation), {'size': x_label_scale})
    plt.figure(figsize=(8, 5.5))
    plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_validation*10)), {'size': x_label_scale})
    plt.ylabel('Gap to CP-SAT', {'size': y_label_scale})
    plt.grid()
    x = np.array([i + 1 for i in range(obj_last_step1.shape[0])])
    plt.plot(x, obj_last_step1, color='tab:blue', label='{}×{}'.format(10, 10))
    plt.plot(x, obj_last_step2, color='tab:red', label='{}×{}'.format(15, 10))
    plt.plot(x, obj_last_step3, color='tab:green', label='{}×{}'.format(15, 15))
    plt.plot(x, obj_last_step4, color='tab:brown', label='{}×{}'.format(20, 10))
    plt.plot(x, obj_last_step5, color='tab:orange', label='{}×{}'.format(20, 15))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('merged_last-step_validation_log', save_file_type))
    if show:
        plt.show()



