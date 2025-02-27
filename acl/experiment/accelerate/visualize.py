import numpy as np
from matplotlib import pyplot as plt

# FT version

fp16_7b_tps = 50.589877111129
fp16_13b_tps = 27.3062883732963

bit = np.array([2.5, 3, 3.5])
bitstack_7b_fused_2_tps = [14.9323422623486, 12.8591301594179, 11.3981001385257]
bitstack_7b_fused_3_tps = [23.1621676163865, 21.0408419716486, 19.1047497518498]
# owq_7b_25bit_tps = [0, 0, 86.4166249267124]
AMQ_7b_tps = [92.29496693, 95.90507391, 104.6325485]

bitstack_13b_fused_2_tps = [10.6262070139438, 9.70397538093318, 8.79238642894422]
bitstack_13b_fused_3_tps = [16.8477179080421, 15.3755665594854, 14.1359968690058]
owq_13b_25bit_tps = [0, 0, 62.7581934258195]
AMQ_13b_tps = [70.42509431, 75.60473393, 82.53214564]

fp16_7b_tps_3090 = 52.92563944
fp16_13b_tps_3090 = 20

bitstack_7b_fused_2_tps_3090 = [11.73500252, 10.56020271, 9.563721254]
bitstack_7b_fused_3_tps_3090 = [12.18367643, 10.90537009, 9.921464711]
AMQ_7b_tps_3090 = [53.73096119, 55.09511646, 59.79723947]

bitstack_13b_fused_2_tps_3090 = [8.785502195, 7.981933248, 7.41109036]
bitstack_13b_fused_3_tps_3090 = [9.839410297, 8.543708086, 7.623698106]
AMQ_13b_tps_3090 = [43.1541442, 42.21750833, 42.64364405]

f = plt.figure(figsize=(20, 4))
ax = f.subplots(ncols=4, nrows=1, )
# ax = f.subplots(ncols=4, nrows=1, )

font = {'size'   : 17}
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

ax[0].grid(True)
ax[1].grid(True)

ax[0].bar(2.5 - 2*width, fp16_7b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[0].bar(bit, bitstack_7b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[0].bar(bit + width, bitstack_7b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax1.bar((bit + 2 * width)[2], owq_7b_25bit_tps[2], width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[0].bar(bit + 2 * width, AMQ_7b_tps, width=width, color=colors[2], edgecolor = 'black')
# ax[0].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[0].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)

ax[1].bar(2.5 - 2*width, fp16_13b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[1].bar(bit, bitstack_13b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[1].bar(bit + width, bitstack_13b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[1].bar(bit + 2 * width, AMQ_13b_tps, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# ax[1].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[1].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
# ax.legend(['C4 PPL', 'Zero-shot Average'], loc='upper center', fontsize=15, ncol=2, bbox_to_anchor=(0.5, 1.315))

ax[0].set_ylim(0, 125)
ax[1].set_ylim(0, 125)

ax[2].grid(True)
ax[3].grid(True)

ax[2].bar(2.5 - 2*width, fp16_7b_tps_3090, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax[2].bar(bit, bitstack_7b_fused_2_tps_3090, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[2].bar(bit + width, bitstack_7b_fused_3_tps_3090, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax1.bar((bit + 2 * width)[2], owq_7b_25bit_tps[2], width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[2].bar(bit + 2 * width, AMQ_7b_tps_3090, width=width, color=colors[2], edgecolor = 'black')
# ax[2].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[2].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[2].tick_params(axis='x', labelsize=20)
ax[2].tick_params(axis='y', labelsize=20)

ax[3].bar(2.5 - 2*width, fp16_13b_tps_3090, width=width, fill = False, label='FP16', edgecolor = 'red', linestyle='--', linewidth=1)
# ax[3].text(3.33, 73.5, '6GB', color = 'green', fontdict={'size'   : 20, 'fontweight':'bold'})
ax[3].text(2.44 - 2*width, 2, 'OOM', color = 'red', rotation = 90, fontdict={'size'   : 20, 'fontweight':'bold'})
ax[3].bar(bit, bitstack_13b_fused_2_tps_3090, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax[3].bar(bit + width, bitstack_13b_fused_3_tps_3090, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax[3].bar(bit + 2 * width, AMQ_13b_tps_3090, width=width, color=colors[2], label='AMQ', edgecolor = 'black')
# ax[3].set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax[3].set_xticks([2.25, 2.5 + 1 * width, 3 + 1 * width, 3.5 + 1 * width], ['16', '2.5', '3', '3.5'])
ax[3].tick_params(axis='x', labelsize=20)
ax[3].tick_params(axis='y', labelsize=20)
# ax[1].legend(loc='upper left')

ax[2].set_ylim(0, 80)
ax[3].set_ylim(0, 80)



# ax[1].legend(loc='upper left', ncol=4, bbox_to_anchor=(0.5, 1.15))

plt.savefig('source_6.png')