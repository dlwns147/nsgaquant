import numpy as np
from matplotlib import pyplot as plt

# FT version

fp16_7b_tps = 50.59
fp16_13b_tps = 27.31 

bit = np.array([2, 3])
memory_7b = np.array([2238, 3010]) / 1024
memory_13b = np.array([4029, 5542]) / 1024

group_wise_7b_tps = [4.58, 6.65]
group_wise_13b_tps = [3.27, 2.93]

layer_wise_7b_tps = [92.92, 94.82]
layer_wise_13b_tps = [76.02, 76.08]

font = {'size'   : 20}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

colors = [
    '#FF6663',
    '#939393',
    '#C83C04',
    '#378375',
    '#6699FF',
    '#FACA00',
    '#2351AB',
]

f = plt.figure(figsize=(10, 5))
ax = f.subplots(ncols=2, nrows=1)

width = 0.3

ax[0].grid(True)
ax[1].grid(True)

ax[0].bar(memory_7b, group_wise_7b_tps, width=width, label='Group-wise', color=colors[3])
ax[0].bar(memory_7b + 1 * width, layer_wise_7b_tps, width=width, label='Layer-wise', color=colors[2])
ax[0].axhline(fp16_7b_tps, color='black', linestyle='--', linewidth = 4, label='FP16')

width = 0.6
ax[1].bar(memory_13b, group_wise_13b_tps, width=width, label='Group-wise', color=colors[3])
ax[1].bar(memory_13b + 1 * width, layer_wise_13b_tps, width=width, label='Layer-wise', color=colors[2])
ax[1].axhline(fp16_13b_tps, color='black', linestyle='--', linewidth = 4, label='FP16')

ax[0].set_ylim(0, 100)
ax[1].set_ylim(0, 100)

ax[0].set_xticks(memory_7b + 0.23 * width, list(map(lambda x: f'{x:.2f}', memory_7b)))
ax[1].set_xticks(memory_13b + 0.5 * width, list(map(lambda x: f'{x:.2f}', memory_13b)))

ax[0].legend(loc='upper left', ncol=4, bbox_to_anchor=(0.014, 1.24))

f.subplots_adjust(top=0.83, bottom=0.1)

# ax[0].set_ylabel('Token/s', fontsize=20)

# ax[3].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
# ax[3].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])

# ax[3].tick_params(axis='x', labelsize=20)
# ax[3].tick_params(axis='y', labelsize=20)
# # ax[1].legend(loc='upper left')

# ax[2].set_ylim(0, 80)
# ax[3].set_ylim(0, 80)



# ax[1].legend(loc='upper left', ncol=4, bbox_to_anchor=(0.5, 1.15))

plt.savefig('group_layer_wise_speed.png')