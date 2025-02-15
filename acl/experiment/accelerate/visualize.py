import numpy as np
from matplotlib import pyplot as plt

# FT version

fp16_7b_tps = 50.589877111129
fp16_13b_tps = 27.3062883732963

bit = np.array([2.5, 3, 3.5])
bitstack_7b_fused_2_tps = [14.9323422623486, 12.8591301594179, 11.3981001385257]
bitstack_7b_fused_3_tps = [23.1621676163865, 21.0408419716486, 19.1047497518498]
# owq_7b_25bit_tps = [0, 0, 86.4166249267124]
AutoLLMQuant_7b_tps = [93.7335040816505, 96.9731523888586, 121.811362536971]

bitstack_13b_fused_2_tps = [10.6262070139438, 9.70397538093318, 8.79238642894422]
bitstack_13b_fused_3_tps = [16.8477179080421, 15.3755665594854, 14.1359968690058]
owq_13b_25bit_tps = [0, 0, 62.7581934258195]
AutoLLMQuant_13b_tps = [72.7342514751956, 77.1233780174684, 86.9166109815997]

f = plt.figure(figsize=(10, 7))
ax1, ax2 = f.subplots(ncols=2, nrows=1)

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

ax1.grid(True)
ax2.grid(True)

ax1.bar(2.5 - 2*width, fp16_7b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax1.bar(bit, bitstack_7b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax1.bar(bit + width, bitstack_7b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax1.bar((bit + 2 * width)[2], owq_7b_25bit_tps[2], width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax1.bar(bit + 2 * width, AutoLLMQuant_7b_tps, width=width, color=colors[2], edgecolor = 'black')
# ax1.bar(bit[2] + 3 * width, AutoLLMQuant_7b_tps[2], width=width, color=colors[2], label='AutoLLMQuant', edgecolor = 'black')

# ax1.bar(bit + 3 * width, AutoLLMQuant_7b_tps, width=width, color=colors[2], label='AutoLLMQuant', edgecolor = 'black')

# ax1.set_xticks(bit + 1.5 * width, ['2.5', '3', '3.5'])
ax1.set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

ax2.bar(2.5 - 2*width, fp16_13b_tps, width=width, color=colors[1], label='FP16', edgecolor = 'black')
ax2.bar(bit, bitstack_13b_fused_2_tps, width=width, color=colors[6], label='BitStack-Fused-2', edgecolor = 'black')
ax2.bar(bit + width, bitstack_13b_fused_3_tps, width=width, color=colors[4], label='BitStack-Fused-3', edgecolor = 'black')
# ax2.bar(bit + 2 * width, owq_13b_25bit_tps, width=width, color=colors[3], label='OWQ', edgecolor = 'black')
ax2.bar(bit + 2 * width, AutoLLMQuant_13b_tps, width=width, color=colors[2], label='AutoLLMQuant', edgecolor = 'black')
ax2.set_xticks(bit + 1 * width, ['2.5', '3', '3.5'])
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

# ax1.set_title('7B')
# ax1.set_xlabel('Bit')
# ax1.set_ylabel('Token/s')
ax2.legend(loc='upper left')

ax1.set_ylim(0, 125)
ax2.set_ylim(0, 125)

plt.savefig('source_8.png')