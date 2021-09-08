import numpy as np
import matplotlib.pyplot as plt

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
show = True
save = False
save_file_type = '.pdf'

message_passing_cpu = np.array([0.25, 1.32, 2.61])
message_passing_gpu = np.array([0.11, 0.15, 0.17])
CPM = np.array([0.34, 11.03, 22.21])

# plotting...
plt.xlabel('Batch Size', {'size': x_label_scale})
plt.ylabel('Total Time (in seconds)', {'size': y_label_scale})
plt.grid()
x = np.array(['1', '64', '128'])
plt.plot(x, message_passing_cpu, color='tab:blue', label='Message-passing CPU')
plt.plot(x, message_passing_gpu, color='tab:red', label='Message-passing GPU')
plt.plot(x, CPM, color='tab:green', label='CPM')
plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
if show:
    plt.show()