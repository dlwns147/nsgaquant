import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info

# ppl_dataset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_ppl_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
# nsga_layer_ppl_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_layer_prune_0.5_1_loss.json'
sleb_128_ppl_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_128_js.csv'
sleb_256_ppl_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_256_js.csv'

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/layer_pruning_result.png'

model_name='meta-llama/Llama-2-7b-hf'

# config='/NAS/SJ/nsgaquant/config/llama.json'
# with open(config, 'r') as f:
#     config = json.load(f)[model_name]

# with open(ppl_dataset_path, 'r') as json_file:
#     ppl_dataset = json.load(json_file)['archive']
#     ppl_dataset_ppl = [d[1] for d in ppl_dataset]
#     ppl_dataset_bits = [d[2] for d in ppl_dataset]

with open(sleb_128_ppl_path, 'r') as csv_file:
    sleb_128_ppl_result = list(csv.reader(csv_file))
    sleb_128_ppl = list(map(float, sleb_128_ppl_result[1]))

with open(sleb_256_ppl_path, 'r') as csv_file:
    sleb_256_ppl_result = list(csv.reader(csv_file))
    sleb_256_ppl = list(map(float, sleb_256_ppl_result[1]))

nsga_layer_ppl = [5.535838127, 5.581321716, 5.666895866, 5.779502392, 5.883149624, 6.048220634, 6.174895287, 6.340240955, 6.552864552, 6.798971653, 7.048149109, 7.343518257, 7.572817802, 7.924718857, 8.292881012, 8.699520111, 9.144404411, 9.714840889, 10.42185402, 11.05092335, 12.20806026, 13.03364563, 15.66242599, 16.64489174, 18.06842422, 20.23059273, 21.68511963, 24.81386185, 27.04511833, 30.92720222, 35.01128387, 42.45939636]
sparsity = [i/64 for i in range(63, 31,-1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0].plot(sparsity, sleb_128_ppl, label='Greedy Search 128 (1200s)')
axes[0].plot(sparsity, sleb_256_ppl, label='Greedy Search 256 (2800s)')
axes[0].plot(sparsity, nsga_layer_ppl, label='NSGA2 (2400s)')
axes[0].set_title('PPL')
axes[0].set_xlabel('Sparsity')
axes[0].set_ylabel('PPL')
axes[0].legend(loc='upper right')
axes[0].set_xlim([0.7, 1.0])
axes[0].set_ylim([5.4, 11])

axes[1].plot(sparsity, sleb_128_ppl, label='Greedy Search 128 (1200s)')
axes[1].plot(sparsity, sleb_256_ppl, label='Greedy Search 256 (2800s)')
axes[1].plot(sparsity, nsga_layer_ppl, label='NSGA2 (2400s)')
# axes[1].scatter(greedy_bits, greedy_loss, color='r', s=3, label='Greedy search')
axes[1].set_title('PPL')
axes[1].set_xlabel('Sparsity')
axes[1].set_ylabel('PPL')
axes[1].legend(loc='upper right')
# axes[1].set_xlim([2.95, 3.05])
# axes[1].set_ylim([2, 3.5])

plt.show()
plt.savefig(ppl_arch_figure, dpi=300)

# ppl_outliers_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_1000_with_outlier.json'
# with open(ppl_outliers_path, 'r') as json_file:
#     dataset = json.load(json_file)

# arch = [np.fromstring(arch, sep=' ', dtype=int) for arch in list(dataset.keys())]
# arch_mean = [a.mean() for a in arch]
# ppl = list(dataset.values())

# plt.scatter(arch_mean, ppl, s=5)
# plt.title('PPL with outliers')
# plt.xlabel('4-bit selection ratio')
# plt.ylabel('PPL')
# plt.show()
# plt.savefig('outlier_dist.png', dpi=300)