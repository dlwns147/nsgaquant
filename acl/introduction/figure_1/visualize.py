import numpy as np
from matplotlib import pyplot as plt

quantSEA_13b_bit = [2.104545455, 2.20309917, 2.40061983, 2.50433884, 2.60392562, 2.74938017, 2.80206612, 3.00165289, 3.20392562, 3.40123967, 3.50061983, 3.60392562, 3.75475207, 3.80041322]
quantSEA_13b_avg = np.array([0.641908905, 0.66219152, 0.682863029, 0.689958569, 0.691083918, 0.703693136, 0.710422927, 0.715917286, 0.71687743, 0.725760049, 0.724154011, 0.722983757, 0.728700119, 0.727874139]) * 100

owq_13b_bit = [2.2,2.4,2.5,2.6,2.8,3.2,3.4,3.5,3.6,3.8]
owq_13b_avg = np.array([0.649320442, 0.64854396, 0.651275389, 0.652410135, 0.660048735, 0.718132311, 0.716591034, 0.720321308, 0.724292107, 0.719481134]) * 100

pb_llm_13b_bit = [2.001, 2.197, 2.4, 2.498, 2.603, 2.799, 3.002, 3.198, 3.401, 3.499, 3.597, 3.8, 4.003]
pb_llm_13b_avg = np.array([0.411433171, 0.445306741, 0.521918144, 0.582221528, 0.608104881, 0.64488449, 0.671657683, 0.683852428, 0.695484065, 0.706756403, 0.708052548, 0.717220212, 0.713828894]) * 100

bitstack_13b_bit = [2.2, 2.4, 2.5, 2.6, 2.8, 3, 3.2, 3.4, 3.5, 3.6, 3.8, 4]
bitstack_13b_avg = np.array([0.646154821, 0.671499793, 0.681265789, 0.687761946, 0.693403778, 0.704266588, 0.708421695, 0.714242304, 0.714063108, 0.715914363, 0.720499593, 0.719499094]) * 100

# group size = 128, symmetric
awq_13b_bit = [2, 3, 4]
awq_13b_avg = np.array([0.393870517, 0.679394636, 0.724932406]) * 100

gptq_13b_bit = [2, 3, 4]
gptq_13b_avg = np.array([0.370548899, 0.673860776, 0.718537608]) * 100

memory = [2315, 2392, 2469, 2546, 2623, 2701, 2778, 2855, 2932, 3009, 3087, 3164, 3241, 3318, 3395, 3473, 3550, 3627, 3704, 3781, 2816, 3588]

f = plt.figure(figsize=(10, 7))
ax1 = f.subplots(ncols=1, nrows=2)

font = {'size'   : 20}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

colors = [
    '#939393',
    '#2351AB',
    '#C83C04',
    '#FACA00',
    '#378375',
    '#6699FF'
]

# calculate memory
one_bit_memory = 1512
scale_zero_memory = 370
embed_head_memory = 635
quantSEA_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in quantSEA_13b_bit]
owq_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in owq_13b_bit]
pb_llm_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in pb_llm_13b_bit]
bitstack_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in bitstack_13b_bit]
awq_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in awq_13b_bit]
gptq_13b_memory = [bit * one_bit_memory + scale_zero_memory + embed_head_memory for bit in gptq_13b_bit]


ax1[0].plot(quantSEA_13b_memory, quantSEA_13b_avg, label='AutoLLMQuant', marker='o', markersize=8, color='r', linewidth=4, zorder=10)
# ax1[0].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=8, color=colors[1], linewidth=4, zorder=1)
ax1[0].plot(pb_llm_13b_memory, pb_llm_13b_avg, label='PB-LLM', marker='o', markersize=8, color=colors[2], linewidth=4)
ax1[0].plot(bitstack_13b_memory, bitstack_13b_avg, label='BitStack', marker='o', markersize=8, color='orange', linewidth=4, zorder = 1)

ax1[1].plot(quantSEA_13b_memory, quantSEA_13b_avg, label='AutoLLMQuant', marker='o', markersize=8, color='r', linewidth=4, zorder=10)
# ax1[1].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=8, color=colors[1], linewidth=4, zorder=1)
ax1[1].plot(pb_llm_13b_memory, pb_llm_13b_avg, label='PB-LLM', marker='o', markersize=8, color=colors[2], linewidth=4)
ax1[1].plot(bitstack_13b_memory, bitstack_13b_avg, label='BitStack', marker='o', markersize=8, color='orange', linewidth=4, zorder=1)

ax1[0].scatter(awq_13b_memory, awq_13b_avg, label='AWQ', s=60, color=colors[5], marker='o', zorder=3)
ax1[0].scatter(gptq_13b_memory, gptq_13b_avg, label='GPTQ', s=60, color=colors[4], marker='o', zorder=3)

ax1[1].scatter(awq_13b_memory, awq_13b_avg, label='AWQ', s=60, color=colors[5], marker='o', zorder=3)
ax1[1].scatter(gptq_13b_memory, gptq_13b_avg, label='GPTQ', s=60, color=colors[4], marker='o', zorder=3)

# plt.xlabel('Bit')
# plt.ylabel('Zero-shot Average')

ax1[0].set_ylim(63, 75)
ax1[1].set_ylim(36, 48)
ax1[0].set_yticks(np.arange(63, 76, 2))
ax1[1].set_yticks(np.arange(36, 49, 2))

# yticks, xticks 크기 늘리기
ax1[0].tick_params(axis='both', which='major', labelsize=20)
ax1[1].tick_params(axis='both', which='major', labelsize=20)

# plt.tight_layout()
# ax1[1].legend(['QuantSEA', 'OWQ', 'PB-LLM', 'BitStack', 'AWQ', 'GPTQ'], loc='lower right')
ax1[1].legend(['AutoLLMQuant', 'PB-LLM', 'BitStack', 'AWQ', 'GPTQ'], loc='lower right')
ax1[0].grid()
ax1[1].grid()
plt.savefig('source_1.png')