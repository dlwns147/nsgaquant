import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots_adjust

# FT version

fp16_7b_tps = 50.589877111129
fp16_13b_tps = 27.3062883732963

bit = np.array([2.5, 3, 3.5])
bitstack_7b_fused_2_tps = [15.4149608, 13.62765656, 12.14004663]
bitstack_7b_fused_3_tps = [24.27706541, 22.02350172, 20.11785806, ]
# owq_7b_25bit_tps = [0, 0, 86.4166249267124]
AMQ_7b_tps = [92.29496693, 95.90507391, 104.6325485]

bitstack_13b_fused_2_tps = [7.15578038, 6.273436638, 5.496908988]
bitstack_13b_fused_3_tps = [11.89239396, 10.66521603, 9.800740556]
owq_13b_25bit_tps = [0, 0, 62.7581934258195]
AMQ_13b_tps = [70.42509431, 75.60473393, 82.53214564]

fp16_7b_tps_3090 = 52.92563944
fp16_13b_tps_3090 = 0

bitstack_7b_fused_2_tps_3090 = [6.865133394, 6.286844307, 5.770865273]
bitstack_7b_fused_3_tps_3090 = [7.504872662, 11.50135343, 10.43382613]
AMQ_7b_tps_3090 = [53.73096119, 55.09511646, 59.79723947]

bitstack_13b_fused_2_tps_3090 = [6.459495215, 3.463286822, 3.034208622]
bitstack_13b_fused_3_tps_3090 = [4.325230686, 3.753797829, 5.003392124447116]
AMQ_13b_tps_3090 = [43.1541442, 42.21750833, 42.64364405]

# f = plt.figure()
# ax = f.subplots(, )
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))

subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.2)

font = {'size'   : 15}
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

width = 0.125

ax[0, 0].grid(True)
ax[1, 0].grid(True)

ax[0, 0].set_title('L40S / Llama 2 7B')
ax[0, 0].bar(2.5 - 2*width, fp16_7b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[0, 0].bar(bit, bitstack_7b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[0, 0].bar(bit + width, bitstack_7b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax1.bar((bit + 2 * width)[2], owq_7b_25bit_tps[2], width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[0, 0].bar(bit + 2 * width, AMQ_7b_tps, width=width, color=colors[2], label = 'AMQ', edgecolor = 'black')
# ax[0, 0].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[0, 0].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[0, 0].tick_params(axis='x', labelsize=20)
ax[0, 0].tick_params(axis='y', labelsize=20)

ax[1, 0].set_title('L40S / Llama 2 13B')
ax[1, 0].bar(2.5 - 2*width, fp16_13b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[1, 0].bar(bit, bitstack_13b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[1, 0].bar(bit + width, bitstack_13b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[1, 0].bar(bit + 2 * width, AMQ_13b_tps, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# ax[1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[1, 0].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[1, 0].tick_params(axis='x', labelsize=20)
ax[1, 0].tick_params(axis='y', labelsize=20)
# ax.legend(['C4 PPL', 'Zero-shot Average'], loc='upper center', fontsize=15, ncol=2, bbox_to_anchor=(0.5, 1.315))

ax[0, 0].set_ylim(0, 125)
ax[1, 0].set_ylim(0, 125)

ax[0, 1].grid(True)
ax[1, 1].grid(True)
ax[0, 0].get_xaxis().set_visible(False)

ax[0, 1].set_ylim(0, 80)

ax[0, 1].set_title('RTX3090 / Llama 2 7B')
ax[0, 1].bar(2.5 - 2*width, fp16_7b_tps_3090, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[0, 1].bar(bit, bitstack_7b_fused_2_tps_3090, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[0, 1].bar(bit + width, bitstack_7b_fused_3_tps_3090, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax1.bar((bit + 2 * width)[2], owq_7b_25bit_tps[2], width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[0, 1].bar(bit + 2 * width, AMQ_7b_tps_3090, width=width, color=colors[2], edgecolor = 'black')
# ax[0, 1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[0, 1].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[0, 1].tick_params(axis='x', labelsize=20)
ax[0, 1].tick_params(axis='y', labelsize=20)

ax[1, 1].set_title('RTX3090 / Llama 2 13B')
ax[1, 1].bar(2.5 - 2*width, fp16_13b_tps_3090, width=width, fill = False, label='FP16', edgecolor = 'red', linestyle='--', linewidth=1)
# ax[1, 1].text(3.33, 73.5, '6GB', color = 'green', fontdict={'size'   : 20, 'fontweight':'bold'})
ax[1, 1].text(2.44 - 2*width, 5, 'OOM', color = 'red', rotation = 90, fontdict={'size'   : 20, 'fontweight':'bold'})
ax[1, 1].bar(bit, bitstack_13b_fused_2_tps_3090, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[1, 1].bar(bit + width, bitstack_13b_fused_3_tps_3090, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[1, 1].bar(bit + 2 * width, AMQ_13b_tps_3090, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# ax[1, 1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[1, 1].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[1, 1].tick_params(axis='x', labelsize=20)
ax[1, 1].tick_params(axis='y', labelsize=20)
# ax[1].legend(loc='upper left')

ax[0, 1].set_ylim(0, 80)
ax[1, 1].set_ylim(0, 80)
ax[0, 1].get_xaxis().set_visible(False)

ax[0, 0].set_ylim(0, 120)
ax[0, 1].set_ylim(0, 120)
ax[0, 1].set_yticklabels([])
ax[0 ,1].grid(True)

ax[1, 0].set_ylim(0, 100)
ax[1, 1].set_ylim(0, 100)
ax[1, 1].set_yticklabels([])
ax[1 ,1].grid(True)


fig.subplots_adjust(top=0.85)
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), labelspacing=0., fontsize=15)
# ax[1].legend(loc='upper left', ncol=4, bbox_to_anchor=(0.5, 1.15))

plt.savefig('/NAS/SJ/nsgaquant/acl/experiment/accelerate/accelerate.png')