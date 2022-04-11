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
embedding_type = 'gin+dghan'
heads = 1
drop_out = 0.

# training parameters
lr = 4e-5
steps_learn = 10
transit = 500
batch_size = 64
episodes = 256000
step_validation = 10

# plot parameters
x_label_scale = 16
y_label_scale = 16
anchor_text_size = 16
total_plt_steps = 400
show = True
save = True
plot_step_size_training = (episodes // batch_size) // total_plt_steps
plot_step_size_validation = (episodes // batch_size) // (total_plt_steps * 10)
save_file_type = '.pdf'


file = '{}x{}[{},{}]_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}_' \
        '{}_{}_{}_{}_{}_{}' \
    .format(j, m, l, h, init_type, reward_type, gamma,
            hidden_dim, embedding_layer, policy_layer, embedding_type, heads, drop_out,
            lr, steps_learn, transit, batch_size, episodes, step_validation)

log1 = np.load('../log/'
               '{}_log_'
               .format('training')  # log type
               + file + '.npy')

log2 = np.load('../log/'
               '{}_log_'
               .format('validation')  # log type
               + file + '.npy')[:, 1]


obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
obj2 = log2[:log2.shape[0] // plot_step_size_validation * plot_step_size_validation].reshape(log2.shape[0] // plot_step_size_validation, -1).mean(axis=1)


# plotting...
# plt.xlabel('Iteration(stride-{})'.format(plot_step_size_training), {'size': x_label_scale})
plt.figure(figsize=(8, 2.5))
plt.xlabel('Every {} batches.'.format(r'${}$'.format(plot_step_size_training)), {'size': x_label_scale})
plt.ylabel('Makespan', {'size': y_label_scale})
plt.grid()
x = np.array([i + 1 for i in range(obj1.shape[0])])
plt.plot(x, obj1, color='#f19000', label='Training', linewidth=1)
plt.plot(x, obj2, color='#008080', label='Validation', linewidth=1)
plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./{}{}'.format('overfitting_study', save_file_type))
if show:
    plt.show()




