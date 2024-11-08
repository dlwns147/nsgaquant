import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/vis_234_results.png'

model_name='meta-llama/Llama-2-7b-hf'

nsga_24_ppl = [5.918757439, 6.308952332, 6.907948494, 7.69556284, 8.753380775, 10.45113373, 14.50975132]
nsga_234_ppl = [5.811964035, 5.951051235, 6.172879219, 6.465126991, 7.138121128, 8.756361961, 12.21780586]
bits = [3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0].scatter(bits, nsga_24_ppl, s=5, label='NSGA 2/4-bits 128gs')
axes[0].scatter(bits, nsga_234_ppl, s=5, label='NSGA 2/3/4-bits 128gs')
axes[0].scatter([3], [7.99], s=5, label='HQQ 3bits 128gs')
axes[0].set_title('PPL')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('PPL')
axes[0].legend(loc='upper right')

# axes[1].plot(sparsity, sleb_128_ppl, label='Greedy Search 128 (1200s)')
# axes[1].plot(sparsity, sleb_256_ppl, label='Greedy Search 256 (2800s)')
# axes[1].plot(sparsity, nsga_layer_ppl, label='NSGA2 (2400s)')
# # axes[1].scatter(greedy_bits, greedy_loss, color='r', s=3, label='Greedy search')
# axes[1].set_title('PPL')
# axes[1].set_xlabel('Sparsity')
# axes[1].set_ylabel('PPL')
# axes[1].legend(loc='upper right')
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