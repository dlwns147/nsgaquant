import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info


# model_path = 'meta-llama'
# model_name='Llama-2-7b-hf'

ppl_arch_figure = f'/NAS/SJ/nsgaquant/fig/awq_results.png'

awq_llama2_7b_nsga_ppl = [5.74, 5.82, 5.95, 6.18, 6.88, 8.09, 10.38]
awq_llama2_13b_nsga_ppl = [5.060882092, 5.118550301, 5.208967686, 5.353271008, 5.698880196, 6.37257576, 7.591520786]
bits = [3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# axes[0].scatter(bits, nsga_24_ppl, s=5, label='NSGA 2/4-bits 128gs')
axes[0].plot(bits, awq_llama2_7b_nsga_ppl, label='NSGA 2/3/4-bits 128gs', color='b')
axes[0].scatter([3], [6.25], s=7, label='AWQ 128gs', color='r')
axes[0].scatter([3], [7.16], s=7, label='LLM-MQ', color='teal')
axes[0].scatter([3], [6.14], s=7, label='CMPQ', color='gray')
axes[0].set_title('Llama-2-7b')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('PPL')
axes[0].legend(loc='upper right')

axes[1].plot(bits, awq_llama2_13b_nsga_ppl, label='NSGA 2/3/4-bits 128gs', color='b')
axes[1].scatter([3], [5.32], s=7, label='AWQ 128gs', color='r')
axes[1].scatter([3], [5.89], s=7, label='LLM-MQ', color='teal')
axes[1].scatter([3], [5.34], s=7, label='CMPQ', color='gray')
axes[1].set_title('Llama-2-13b')
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('PPL')
axes[1].legend(loc='upper right')

# ppl_arch_figure = f'/NAS/SJ/nsgaquant/fig/hqq_results.png'

# hqq_llama2_7b_nsga_ppl = [5.74, 5.80, 5.91, 6.06, 6.34, 7.08, 8.65, 12.14]
# hqq_llama2_13b_nsga_ppl = [4.99743509292602, 5.047353268, 5.122300625, 5.243927002, 5.41167593, 5.852484226, 6.700005531, 8.389808655]
# bits = [4, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25]

# axes[0].plot(bits, hqq_llama2_7b_nsga_ppl, label='NSGA 2/3/4-bits 128gs', color='b')
# axes[0].scatter([3, 4], [7.99, 5.74], s=7, label='HQQ 128gs', color='r')
# axes[0].scatter([3], [7.16], s=7, label='LLM-MQ', color='teal')
# axes[0].scatter([3], [6.14], s=7, label='CMPQ', color='gray')
# axes[0].set_title('Llama-2-7b')
# axes[0].set_xlabel('Bits')
# axes[0].set_ylabel('PPL')
# axes[0].legend(loc='upper right')

# axes[1].plot(bits, hqq_llama2_13b_nsga_ppl, label='NSGA 2/3/4-bits 128gs', color='b')
# axes[1].scatter([3, 4], [5.86, 4.99743509292602], s=7, label='HQQ 128gs', color='r')
# axes[1].scatter([3], [5.89], s=7, label='LLM-MQ', color='teal')
# axes[1].scatter([3], [5.34], s=7, label='CMPQ', color='gray')
# axes[1].set_title('Llama-2-13b')
# axes[1].set_xlabel('Bits')
# axes[1].set_ylabel('PPL')
# axes[1].legend(loc='upper right')

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