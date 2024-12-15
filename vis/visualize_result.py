import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/result.png'

model_name='meta-llama/Llama-2-7b-hf'

# greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'
# greedy_ppl_reverse_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_reverse_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'

greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_128_sqs_false_sqz_false.csv'

config='/NAS/SJ/nsgaquant/config/llama.json'
with open(config, 'r') as f:
    config = json.load(f)[model_name]

greedy_outlier_idx = 5
with open(greedy_ppl_path, 'r') as csv_file:
    greedy_ppl_result = list(csv.reader(csv_file))
    greedy_bits = list(map(float, greedy_ppl_result[1]))[:-greedy_outlier_idx]
    greedy_ppl = list(map(float, greedy_ppl_result[2]))[:-greedy_outlier_idx]

# with open(greedy_ppl_reverse_path, 'r') as csv_file:
#     greedy_reverse_ppl_result = list(csv.reader(csv_file))
#     greedy_reverse_bits = list(map(float, greedy_reverse_ppl_result[1]))[greedy_outlier_idx:]
#     greedy_reverse_ppl = list(map(float, greedy_reverse_ppl_result[2]))[greedy_outlier_idx:]

# bits = [3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2]
bits = ['Total', 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
iter_300_bits = [3.7539, 3.5006, 3.2471, 3.0019, 2.7519, 2.5032, 2.2513]
iter_300_ppl = [5.918757439, 6.308952332, 6.907948494, 7.69556284, 8.753380775, 10.45113373, 14.50975132]


# iter_200_bits = [3.7523, 3.5006, 3.249, 2.9971, 2.7503, 2.4997, 2.2529]
# iter_200_ppl = [5.895573139, 6.182063103, 6.542125702, 7.016082764, 7.678536892, 8.608513832, 10.18992901]

x_lim_min = [2, 2.2, 2.45, 2.7, 2.95, 3.20, 3.45, 3.7]
x_lim_max = [4, 2.3, 2.55, 2.8, 3.05, 3.30, 3.55, 3.8]
y_lim_min = [5, 14, 10, 8.5, 7.5, 6.5, 6.0, 5.75]
y_lim_max = [32, 16, 12, 9.5, 8.5, 7.5, 6.5, 6.25]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, ax in enumerate(axes.flatten()):
    ax.scatter(greedy_bits, greedy_ppl, color='teal', s=3, label='Greedy Search (4->2) (14h)')
    # ax.scatter(greedy_reverse_bits, greedy_reverse_ppl, color='orange', s=3, label='Greedy search (2->4) (14h)')
    ax.scatter(iter_300_bits, iter_300_ppl, color='red', s=3, label='NSGA (11h)')
    # ax.scatter(iter_200_bits, iter_200_ppl, color='purple', s=3, label='NSGA (17h)')
    ax.set_title(f'Bits: {bits[i]}')
    ax.set_xlabel('Bits')
    ax.set_ylabel('PPL')
    ax.set_xlim([x_lim_min[i], x_lim_max[i]])
    ax.set_ylim([y_lim_min[i], y_lim_max[i]])
    ax.legend(loc="upper right")

plt.show()
plt.savefig(ppl_arch_figure, dpi=300)