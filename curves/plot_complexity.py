import numpy as np
import matplotlib.pyplot as plt


methods = ['L2D', 'RL-GNN', 'Ours']

x_labels = ['5', '10', '15', '20', '25', '30']
steps = [500]

fixed = 'm=5'  # 'j=30', 'm=5'


times_for_plot = []

for method in methods:
    if method == 'Ours':
        time = np.load('../complexity/L2S_complexity_{}_{}.npy'.format(fixed, steps))[:6]
    else:
        time = np.load('../complexity/{}_complexity_{}.npy'.format(method, fixed))[:6]
    times_for_plot.append(time)


# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
title_size = 15
show = False
save = True
save_file_type = '.png'


obj1 = times_for_plot[0]  # L2D
obj2 = times_for_plot[1]  # RL-GNN
obj3_1 = times_for_plot[2][:, 0].reshape(-1)  # ours-500
# obj3_2 = times_for_plot[2][:, 1].reshape(-1)  # ours-1000
# obj3_3 = times_for_plot[2][:, 2].reshape(-1)  # ours-1500
# plotting...
# plt.xlabel('Iteration(stride-{})'.format(plot_step_size_training), {'size': x_label_scale})
# plt.title('Computation time of m=5', {'size': title_size})
plt.figure(figsize=(5.3, 5))
if fixed == 'm=5':
    plt.xlabel('Number of jobs {}'.format(r'$n$'), {'size': x_label_scale})
else:
    plt.xlabel('Number of machines {}'.format(r'$m$'), {'size': x_label_scale})

plt.ylabel('Seconds', {'size': y_label_scale})
plt.grid()
x = np.array(x_labels)
# plt.plot(x, obj1, color='tab:green', marker="o", label=methods[0])
plt.plot(x, obj2, color='tab:red', marker="s", label=methods[1])
plt.plot(x, obj3_1, color='tab:blue', linestyle="--", marker="v", label=methods[2] + '-' + str(steps[0]))
# plt.plot(x, obj3_2, color='tab:blue', linestyle="--", marker="^", label=methods[2] + '-' + str(steps[1]))
# plt.plot(x, obj3_3, color='tab:blue', linestyle="--", marker="<", label=methods[2] + '-' + str(steps[2]))
# for i, (xe, ye) in enumerate(zip(x, obj3_1)):
#     if i == 0:
#         plt.plot([xe] * len(ye), ye, '--', color='tab:green', marker="d", label=methods[2])
#     else:
#         plt.plot([xe] * len(ye), ye, '--', color='tab:green', marker="d")
plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./complexity_analysis_{}{}'.format(fixed, save_file_type))
if show:
    plt.show()