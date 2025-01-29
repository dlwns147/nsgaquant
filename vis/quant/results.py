import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/quant/results.png'

# model_name='Llama-2-7b-hf'

# greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'
# greedy_ppl_reverse_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_reverse_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'

greedy_path_7b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_hqq_24bits_ppl_desc_1axis_64_128gs_jsd.csv'
greedy_23_path_7b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_hqq_23bits_ppl_desc_1axis_64_128gs_jsd.csv'
greedy_34_path_7b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_hqq_34bits_ppl_desc_1axis_64_128gs_jsd.csv'
greedy_path_13b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-13b-hf_hqq_24bits_ppl_desc_1axis_64_128gs_jsd.csv'
greedy_23_path_13b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-13b-hf_hqq_23bits_ppl_desc_1axis_64_128gs_jsd.csv'
greedy_34_path_13b = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-13b-hf_hqq_34bits_ppl_desc_1axis_64_128gs_jsd.csv'

# config='/NAS/SJ/nsgaquant/config/llama.json'
# with open(config, 'r') as f:
#     config = json.load(f)[model_name]

greedy_outlier_idx = 30
with open(greedy_path_7b, 'r') as csv_file:
    greedy_result_7b = list(csv.reader(csv_file))
    greedy_bits_7b = list(map(float, greedy_result_7b[1]))[:-greedy_outlier_idx]
    greedy_ppl_7b = list(map(float, greedy_result_7b[2]))[:-greedy_outlier_idx]
    # greedy_bits_7b = [3.7509715025906734,3.50550518134715,3.250323834196891,3.0016191709844557,2.7529145077720205,2.5058290155440415,2.25]
    # greedy_ppl_7b = [5.823429584503174,6.101537227630615,6.581286430358887,7.162560939788818,7.943966865539551,8.894411087036133,10.567473411560059]

with open(greedy_path_13b, 'r') as csv_file:
    greedy_result_13b = list(csv.reader(csv_file))
    greedy_bits_13b = list(map(float, greedy_result_13b[1]))[:-greedy_outlier_idx]
    greedy_ppl_13b = list(map(float, greedy_result_13b[2]))[:-greedy_outlier_idx]

with open(greedy_23_path_7b, 'r') as csv_file:
    greedy_23_result_7b = list(csv.reader(csv_file))
    greedy_23_bits_7b = list(map(float, greedy_23_result_7b[1]))[2:-greedy_outlier_idx * 2]
    greedy_23_ppl_7b = list(map(float, greedy_23_result_7b[2]))[2:-greedy_outlier_idx * 2 ]
    # greedy_23_bits_7b = [2.7490284974093266,2.5030764248704664,2.25]
    # greedy_23_ppl_7b = [8.37462043762207,9.961211204528809,12.950129508972168]

with open(greedy_34_path_7b, 'r') as csv_file:
    greedy_34_result_7b = list(csv.reader(csv_file))
    greedy_34_bits_7b = list(map(float, greedy_34_result_7b[1]))[:-2]
    greedy_34_ppl_7b = list(map(float, greedy_34_result_7b[2]))[:-2]
    # greedy_34_bits_7b = [3.75,3.499190414507772,3.2512953367875648]
    # greedy_34_ppl_7b = [5.785968780517578,5.918093204498291,6.080503463745117]

with open(greedy_23_path_13b, 'r') as csv_file:
    greedy_23_result_13b = list(csv.reader(csv_file))
    greedy_23_bits_13b = list(map(float, greedy_23_result_13b[1]))[2:-greedy_outlier_idx * 2]
    greedy_23_ppl_13b = list(map(float, greedy_23_result_13b[2]))[2:-greedy_outlier_idx * 2 ]

with open(greedy_34_path_13b, 'r') as csv_file:
    greedy_34_result_13b = list(csv.reader(csv_file))
    greedy_34_bits_13b = list(map(float, greedy_34_result_13b[1]))[:-2]
    greedy_34_ppl_13b = list(map(float, greedy_34_result_13b[2]))[:-2]

nsga_234_bits_7b = [3.7518, 3.5008, 3.2528, 3.0019, 2.7545, 2.5026, 2.2531]
nsga_234_ppl_7b = [5.79833746, 5.904635429, 6.063999653, 6.299592972, 6.827088833, 7.811058998, 9.572336197]

nsga_234_bits_13b = [3.7498, 3.4992, 3.251, 3.0033, 2.7541, 2.505, 2.2531]
nsga_234_ppl_13b = [5.045349598, 5.122541904, 5.224804401, 5.383463383, 5.694204807, 6.265073299, 7.245925903]

# nsga_234_outlier_bits_7b = [3.7491, 3.5049, 3.2469, 3.0046, 2.7549, 2.5045, 2.2528]
# nsga_234_outlier_ppl_7b = [5.791144371, 5.896235466, 6.0414958, 6.275548935, 6.842689991, 7.849799156, 9.561811447]

# nsga_234_outlier_bits_13b = [3.7542, 3.5014, 3.2473, 3.0019, 2.753, 2.5005, 2.2522]
# nsga_234_outlier_ppl_13b = [5.028992176, 5.109344482, 5.21629715, 5.359463215, 5.681489944, 6.218622684, 7.083488464]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

axes[0].scatter(nsga_234_bits_7b, nsga_234_ppl_7b, label='Ours', s=5)
# axes[0].scatter(nsga_234_outlier_bits_7b, nsga_234_outlier_ppl_7b, s=5, label='nsga 2/2.1/3/3.1/4', alpha=0.5)
axes[0].scatter([3], [7.99], s=5, label='HQQ', alpha=0.8)
axes[0].scatter([3], [6.45], s=5, label='GPTQ', alpha=0.8)
# axes[0].scatter([3], [6.20], s=5, label='awq', alpha=0.5)
axes[0].scatter(greedy_bits_7b, greedy_ppl_7b, label='IS 2-4', s=5)
axes[0].scatter(greedy_23_bits_7b, greedy_23_ppl_7b, label='IS 2-3', s=5)
axes[0].scatter(greedy_34_bits_7b, greedy_34_ppl_7b, label='IS 3-4', s=5)
axes[0].set_title('Llama 2 7b')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('Wikitext2 PPL')
axes[0].grid(c='0.8')
axes[0].set_axisbelow(True)
# axes[0].set_xlim([2.9, 4.1])
# axes[0].set_ylim([None, 7])
# axes[0].set_ylim([None, 10])
axes[0].legend(loc="upper right")

axes[1].scatter(nsga_234_bits_13b, nsga_234_ppl_13b, s=5, label='Ours', alpha=0.8)
# axes[1].scatter(nsga_234_outlier_bits_13b, nsga_234_outlier_ppl_13b, s=5, label='nsga 2/2.1/3/3.1/4', alpha=0.5)
axes[1].scatter([3], [5.86], label='HQQ', s=5, alpha=0.8)
axes[1].scatter([3], [5.48], s=5, label='GPTQ', alpha=0.8)
# axes[1].scatter([3], [5.31], label='awq', s=5, alpha=0.5)
axes[1].scatter(greedy_bits_13b, greedy_ppl_13b, s=5, label='IS 2-4', alpha=0.8)
axes[1].scatter(greedy_23_bits_13b, greedy_23_ppl_13b, s=5, label='IS 2-3', alpha=0.8)
axes[1].scatter(greedy_34_bits_13b, greedy_34_ppl_13b, s=5, label='IS 3-4', alpha=0.8)
axes[1].set_title('Llama 2 13B')
axes[1].set_xlabel('Bits')
axes[1].grid(c='0.8')
axes[1].set_axisbelow(True)
# axes[1].set_ylabel('PPL')
axes[1].legend(loc="upper right")

# ax.set_xlim([x_lim_min[i], x_lim_max[i]])
# ax.set_ylim([y_lim_min[i], y_lim_max[i]])

plt.show()
plt.savefig(ppl_arch_figure, dpi=300)