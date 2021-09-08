import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# problem config
# 'greedy',
# 'best-improvement',
# 'first-improvement',

# 'incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_2_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin_NAN_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_dghan_1_0.0_5e-05_10_500_64_128000_10'

# 'incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_15x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_15x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_20x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10'
# 'incumbent_20x15[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10'

baseline = ['incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_1_0.0_5e-05_10_500_64_128000_10',
            'incumbent_10x10[1,99]_fdd-divide-mwkr_yaoxin_1_128_4_4_gin+dghan_2_0.0_5e-05_10_500_64_128000_10']

# ['500-steps', '1000-steps', '2000-steps', '2000-steps']
x_labels = ['500-steps', '1000-steps', '2000-steps', '5000-steps']


gaps_for_plot = []

for method in baseline:
    gaps = np.array(pd.read_excel('../test_results/{}_gap.xlsx'.format(method)))[:-1, :][:4, :]
    gaps_for_plot.append(gaps)

gaps_for_plot = np.array(gaps_for_plot).reshape(len(baseline), -1) * 100

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
show = False
save = True
save_file_type = '.pdf'


# plotting...
tpm_cam = gaps_for_plot[0]
tpm = gaps_for_plot[1]
# cam = gaps_for_plot[2]

x = np.arange(len(x_labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - (1/2)*width, tpm_cam, width, label='TPM + CAM-1-Head: 10×10', color='#f19b61')
rects2 = ax.bar(x + (1/2)*width, tpm, width, label='TPM + CAM-2-Head: 10×10', color='#b0c4de')
# rects3 = ax.bar(x + width, cam, width, label='CAM', color='#b0c4de', hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Optimal gaps', {'size': y_label_scale})
ax.set_xlabel('Number of testing steps', {'size': x_label_scale})
plt.grid(axis='y', zorder=0)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.legend(fontsize=anchor_text_size)
ax.set_axisbelow(True)

ax.bar_label(rects1, padding=3, fmt='%.1f%%')
ax.bar_label(rects2, padding=3, fmt='%.1f%%')
# ax.bar_label(rects3, padding=3, fmt='%.1f%%')
# ax.text(s="{}%".format(10), ha='center')

fig.tight_layout()

if save:
    plt.savefig('./{}{}'.format('ablation_study_for_heads', save_file_type))
if show:
    plt.show()

plt.show()