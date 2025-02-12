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

awq_13b_bit = [2, 3, 4]
awq_13b_avg = np.array([0.393870517, 0.679394636, 0.724932406]) * 100

gptq_13b_bit = [2, 3, 4]
gptq_13b_avg = np.array([0.370548899, 0.673860776, 0.718537608]) * 100

plt.figure(figsize=(10, 7))
font = {'size'   : 15}
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

plt.plot(quantSEA_13b_bit, quantSEA_13b_avg, label='quantSEA', marker='o', markersize=5, color='r', linewidth=3)
plt.plot(owq_13b_bit, owq_13b_avg, label='OWQ', marker='o', markersize=5, color=colors[1], linewidth=3)
plt.plot(pb_llm_13b_bit, pb_llm_13b_avg, label='PB-LLM', marker='o', markersize=5, color=colors[2], linewidth=3)
plt.plot(bitstack_13b_bit, bitstack_13b_avg, label='BitStack', marker='o', markersize=5, color='orange', linewidth=3)

plt.scatter(awq_13b_bit, awq_13b_avg, label='AWQ', s=100, color=colors[5], marker='o')
plt.scatter(gptq_13b_bit, gptq_13b_avg, label='GPTQ', s=100, color='black', marker='o')

plt.xlabel('Bit')
plt.ylabel('Zero-shot Average')

plt.legend()
plt.grid()
plt.savefig('figure_1.png')