import numpy as np
from matplotlib import pyplot as plt

bit = np.array([2.5, 3, 3.5])
fp16_13b_tps = 27.3062883732963
bitstack_13b_fused_2_tps = [7.15578038, 6.273436638, 5.496908988]
bitstack_13b_fused_3_tps = [11.89239396, 10.66521603, 9.800740556]
AMQ_13b_tps = [70.42509431, 75.60473393, 82.53214564]

bit = bit[0]
bitstack_13b_fused_2_tps = bitstack_13b_fused_2_tps[0]
bitstack_13b_fused_3_tps = bitstack_13b_fused_3_tps[0]
AMQ_13b_tps = AMQ_13b_tps[0]

colors = [
    '#FF6663',
    '#939393',
    '#C83C04',
    '#378375',
    '#6699FF',
    '#FACA00',
    '#2351AB',
]

font = {'size'   : 20}
plt.rc('font', **font)

width = 0.125

f = plt.figure(figsize=(4, 8))
ax = f.subplots(ncols=1, nrows=1)

# ax.set_title('L40S / Llama 2 13B')
# ax.bar(2.5 - 2*width, fp16_13b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
# ax.bar(bit, bitstack_13b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
# ax.bar(bit + width, bitstack_13b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# # ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
# ax.bar(bit + 2 * width, AMQ_13b_tps, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# # ax[1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
# ax.set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)

# ax.set_title('L40S / Llama 2 13B')
ax.bar(2.5 - 2*width, fp16_13b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax.bar(bit, bitstack_13b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax.bar(bit + width, bitstack_13b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax.bar(bit + 2 * width, AMQ_13b_tps, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# ax[1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax.set_xticks([2.25, 2.5 + 1 * width], ['16', '2.5'])
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.grid()
# ax.set_frame_on(False)
ax.legend(loc='upper left', fontsize=13)

plt.savefig('speed_13b.png', dpi=500)