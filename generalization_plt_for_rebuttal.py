import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ours_500 = np.array([0.045765, 0.046491, 0.044445, 0.107552]) * 100
ours_1000 = np.array([0.038423, 0.038919, 0.037269, 0.095060]) * 100
ours_2000 = np.array([0.030924, 0.034111, 0.031467, 0.077821]) * 100
ours_5000 = np.array([0.024598, 0.027372, 0.023298, 0.064257]) * 100

ours = np.stack([ours_500, ours_1000, ours_2000, ours_5000])
taillard, large_pt, gaussian_pt, shared_pc = ours

x_labels = ['Taillard', 'Large-PT', 'Gaussian-PT', 'Shared-PC']

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 10
show = True
save = False
save_file_type = '.pdf'


# plotting...

x = np.arange(len(x_labels)) * 5  # the label locations
width = 1  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 2.5))
# rects1 = ax.bar(x - 3*width, ours_500, width, label='taillard', color='#f19b61')
# rects2 = ax.bar(x - 2*width, ours_1000, width, label='large_pt', color='#b0c4de')
# rects3 = ax.bar(x + 2*width, ours_2000, width, label='gaussian_pt', color='#8fbc8f')
# rects4 = ax.bar(x + 3*width, ours_5000, width, label='shared_pc', color='blue')

rects1 = ax.bar(x - width, taillard, width, label='500 steps', color='#f19b61')
rects2 = ax.bar(x + 0*width, large_pt, width, label='1000 steps', color='#b0c4de')
rects3 = ax.bar(x + 1*width, gaussian_pt, width, label='2000 steps', color='#8fbc8f')
rects4 = ax.bar(x + 2*width, shared_pc, width, label='5000 steps', color='#d18c8d')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Optimal Gaps', {'size': y_label_scale})
# ax.set_xlabel('Number of testing steps', {'size': x_label_scale})
plt.grid(axis='y', zorder=0)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x + width/2)
ax.set_xticklabels(x_labels, fontsize=13)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.legend(fontsize=anchor_text_size, loc='upper left')
# l2 = ax.legend(fontsize=anchor_text_size, loc='upper left')
# plt.gca().add_artist(l2)
ax.set_axisbelow(True)

ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')
ax.bar_label(rects3, padding=3, fmt='%.1f')
ax.bar_label(rects4, padding=3, fmt='%.1f')
# ax.bar_label(rects3, padding=3, fmt='%.1f%%')
# ax.text(s="{}%".format(10), ha='center')

fig.tight_layout()

if save:
    plt.savefig('./{}{}'.format('generalization', save_file_type))
if show:
    plt.show()
