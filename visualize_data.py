import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info

ppl_dataset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_ppl_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
loss_dataset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_loss_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/dataset.png'

greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'
greedy_loss_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_loss_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'

model_name='meta-llama/Llama-2-7b-hf'

config='/NAS/SJ/nsgaquant/config/llama.json'
with open(config, 'r') as f:
    config = json.load(f)[model_name]

seqlen = 2048

# with open(ppl_binom_path, 'r') as json_file:
#     ppl_binom = json.load(json_file)
with open(ppl_dataset_path, 'r') as json_file:
    ppl_dataset = json.load(json_file)['archive']
    ppl_dataset_ppl = [d[1] for d in ppl_dataset]
    ppl_dataset_bits = [d[2] for d in ppl_dataset]

with open(loss_dataset_path, 'r') as json_file:
    loss_dataset = json.load(json_file)['archive']
    loss_dataset_loss = [d[1] for d in loss_dataset]
    loss_dataset_bits = [d[2] for d in loss_dataset]

greedy_last_idx = -5
with open(greedy_ppl_path, 'r') as csv_file:
    greedy_csv = list(csv.reader(csv_file))
    greedy_bits = list(map(float, greedy_csv[1]))[:greedy_last_idx]
    greedy_ppl = list(map(float, greedy_csv[2]))[:greedy_last_idx]

with open(greedy_loss_path, 'r') as csv_file:
    greedy_csv = list(csv.reader(csv_file))
    greedy_loss = list(map(float, greedy_csv[2]))[:greedy_last_idx]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0].scatter(ppl_dataset_bits, ppl_dataset_ppl, s=3, label='PPL pretrain set')
axes[0].scatter(greedy_bits, greedy_ppl, color='r', s=3, label='Greedy search')
axes[0].set_title('PPL')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('PPL')
axes[0].legend(loc='upper right')
axes[0].set_xlim([2.95, 3.05])
axes[0].set_ylim([5, 15])

axes[1].scatter(loss_dataset_bits, loss_dataset_loss, s=3, label='Loss preatrain set')
axes[1].scatter(greedy_bits, greedy_loss, color='r', s=3, label='Greedy search')
axes[1].set_title('Loss')
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right')
axes[1].set_xlim([2.95, 3.05])
axes[1].set_ylim([2, 3.5])

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