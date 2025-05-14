import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

pb_llm_13b_bit = np.array([2.001, 2.197, 2.253, 2.4, 2.498, 2.603, 2.75, 2.799, 3.002, 3.198, 3.247, 3.401, 3.499, 3.597, 3.8, 4.003]) + 0.25
pb_llm_13b_w2 = [70.201988, 49.662487, 32.704044, 21.767782, 15.748631, 12.027323, 9.572448, 9.540066, 7.770907, 6.957156, 6.786724, 6.444432, 6.197367, 6.03354, 5.794348, 5.621876]
pb_llm_13b_c4 = [76.811501, 53.145523, 42.495216, 28.799479, 21.833565, 16.243132, 13.296763, 13.139219, 10.783956, 9.700974, 9.440514, 8.908892, 8.523313, 8.274727, 7.935522, 7.717345]
pb_llm_13b_avg = np.array([0.411433171, 0.445306741, 0.465469785, 0.521918144, 0.582221528, 0.608104881, 0.634757553, 0.64488449, 0.671657683, 0.683852428, 0.686756827, 0.695484065, 0.706756403, 0.708052548, 0.717220212, 0.713828894]) * 100

owq_13b_bit = np.array([2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8]) * 1512.5 + 378.125 + 625 + 0.791015625
owq_13b_w2 = [8.141159058, 7.90238905, 7.906852245, 7.644672871, 5.259451866, 5.233787537, 5.246657372, 5.222622395]
owq_13b_c4 = [10.81324387, 10.55164051, 10.35543156, 10.24857807, 7.184574127, 7.174242496, 7.165292263, 7.144797802]
owq_13b_avg = np.array([0.649320442, 0.64854396, 0.652410135, 0.660048735, 0.718132311, 0.716591034, 0.724292107, 0.719481134]) * 100

bitstack_13b_bit = np.array([2.1, 2.2, 2.25, 2.4, 2.5, 2.6, 2.75, 2.8, 3, 3.2, 3.25, 3.4, 3.5, 3.6, 3.8, 4]) + 0.25
bitstack_13b_w2 = [8.028745651, 7.592807293, 7.456870079, 6.886278152, 6.674023628, 6.534313202, 6.332436562, 6.283027649, 6.042622089, 5.793498993, 5.760062218, 5.671768665, 5.621623516, 5.578070641, 5.509950161, 5.4714818]
bitstack_13b_c4 = [10.8984251, 10.31787777, 10.13041687, 9.537553787, 9.237815857, 9.025455475, 8.73268795, 8.665048599, 8.353291512, 7.974319458, 7.92961359, 7.801470757, 7.7389431, 7.675413132, 7.572425842, 7.49377346]
bitstack_13b_avg = np.array([0.632752944, 0.646154821, 0.652069036, 0.671499793, 0.681265789, 0.687761946, 0.692470769, 0.693403778, 0.704266588, 0.708421695, 0.708730449, 0.714242304, 0.714063108, 0.715914363, 0.720499593, 0.719499094]) * 100

amq_13b_bit = np.array([2.104545455, 2.20309917, 2.25454545, 2.40061983, 2.50433884, 2.60392562, 2.74938017, 2.80206612, 3.00165289, 3.10371901, 3.20392562, 3.25454545, 3.40123967, 3.50061983, 3.60392562, 3.75475207, 3.80041322, 3.98615702]) + 0.25
amq_13b_w2 = [7.71299839, 7.209764004, 6.960026741, 6.441867828, 6.211341858, 5.988779068, 5.742465019, 5.634430885, 5.374492168, 5.304297924, 5.24823761, 5.228238106, 5.157777786, 5.128776073, 5.090263367, 5.030594826, 5.024534225, 4.965404987]
amq_13b_c4 = [10.42893791, 9.835852623, 9.534765244, 8.829443932, 8.517198563, 8.194261551, 7.803359032, 7.670791626, 7.335062981, 7.241182327, 7.170888901, 7.136387348, 7.054496765, 7.015735626, 6.962110519, 6.897399902, 6.886018276, 6.837729931]
amq_13b_avg = np.array([0.641908905, 0.66219152, 0.665796602, 0.682863029, 0.689958569, 0.691083918, 0.703693136, 0.710422927, 0.715917286, 0.718818017, 0.71687743, 0.72231014, 0.725760049, 0.724154011, 0.722983757, 0.728700119, 0.727874139, 0.728116507])  * 100

# awq_13b_bit = np.array([2, 3, 4]) + 0.25
# awq_13b_avg = '40.58586574  72.1126803  72.89647707'.split()
# awq_13b_avg = [float(i) for i in awq_13b_avg]

# gptq_13b_bit = np.array([2, 3, 4]) + 0.25
# gptq_13b_avg = '45.6424071  70.88767941  72.87781402'.split()
# gptq_13b_avg = [float(i) for i in gptq_13b_avg]

awq_13b_bit = np.array([2.5, 3, 4])
awq_13b_avg = '40.58586574  67.3409126  72.58648748'.split()
awq_13b_avg = [float(i) for i in awq_13b_avg]

gptq_13b_bit = np.array([2.5, 3, 4])
gptq_13b_avg = '45.6424071  67.38607757  71.85376084'.split()
gptq_13b_avg = [float(i) for i in gptq_13b_avg]

amq_13b_235_bit = amq_13b_bit[0]
amq_13b_w2 = 7.71299839
amq_13b_c4 = 10.42893791

pb_llm_13b_235_bit = pb_llm_13b_bit[0]
pb_llm_13b_w2 = 70.201988
pb_llm_13b_c4 = 76.811501

bitstack_13b_235_bit = bitstack_13b_bit[0]
bitstack_13b_w2 = 8.028745651
bitstack_13b_c4 = 10.8984251

awq_13b_225_bit = awq_13b_bit[0]
awq_13b_w2 = 122707.5156
awq_13b_c4 = 95547.30469

gptq_13b_225_bit = gptq_13b_bit[0]
gptq_13b_w2 = 27.7838192
gptq_13b_c4 = 23.3856926

font = {'size'   : 20}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

f = plt.figure(figsize=(10, 4))
ax1 = [f.subplots(ncols=1, nrows=1)]

# gs = f.add_gridspec(2, 2)
# ax1 = [f.add_subplot(gs[0, :]), f.add_subplot(gs[1, 0])]
# ax2 = f.add_subplot(gs[1, 1])

colors = [
    '#939393',
    '#2351AB',
    '#C83C04',
    '#FACA00',
    '#378375',
    '#6699FF'
]

# calculate memory
one_bit_memory = 1512.5
scale_zero_memory = 378.125
embed_head_memory = 625
norm_size = 0.791015625
amq_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in amq_13b_bit]) / 1000
owq_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in owq_13b_bit]) / 1000
pb_llm_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in pb_llm_13b_bit]) / 1000
bitstack_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in bitstack_13b_bit]) / 1000
awq_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in awq_13b_bit]) / 1000
gptq_13b_memory = np.array([(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in gptq_13b_bit]) / 1000

linestyle = '--'
markersize = 8
linewidth = 3
s = 320

def func0():
    ax1[0].plot(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', marker='o', markersize=markersize, color='r', linewidth=linewidth, zorder=10)
    # ax1[0].bar(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', color='grey', width = 50, linewidth=linewidth, zorder=1)
    # ax1[0].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=markersize, color=colors[1], linewidth=linewidth, zorder=1)
    ax1[0].plot(bitstack_13b_memory, bitstack_13b_avg, linestyle = linestyle, label='BitStack', marker='o', markersize=markersize, color='orange', linewidth=linewidth, zorder = 5)
    ax1[0].plot(pb_llm_13b_memory, pb_llm_13b_avg, linestyle = linestyle, label='PB-LLM', marker='o', markersize=markersize, color=colors[4], linewidth=linewidth, zorder=5)

    ax1[0].scatter(awq_13b_memory[-1], awq_13b_avg[-1], label='AWQ', s=s, color='dodgerblue', marker='*', zorder=16)
    ax1[0].scatter(gptq_13b_memory[-1], gptq_13b_avg[-1], label='GPTQ', s=s, color='brown', marker='*', zorder=15)
    
    ax1[0].scatter(awq_13b_memory[1], awq_13b_avg[1], s=s, color='dodgerblue', marker=MarkerStyle('*', fillstyle='right'), zorder=15)
    ax1[0].scatter(gptq_13b_memory[1], gptq_13b_avg[1], s=s, color='brown', marker=MarkerStyle('*', fillstyle='left'), zorder=15)

def func1():
    ax1[1].plot(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', marker='o', markersize=markersize, color='r', linewidth=linewidth, zorder=10)
    # ax1[1].bar(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', color='grey', width = 50, linewidth=linewidth, zorder=1)
    # ax1[1].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=markersize, color=colors[1], linewidth=linewidth, zorder=1)
    ax1[1].plot(pb_llm_13b_memory, pb_llm_13b_avg, linestyle = linestyle, label='PB-LLM', marker='o', markersize=markersize, color=colors[4], linewidth=linewidth)
    ax1[1].plot(bitstack_13b_memory, bitstack_13b_avg, linestyle = linestyle, label='BitStack', marker='o', markersize=markersize, color='orange', linewidth=linewidth, zorder=3)
    
    ax1[1].scatter(awq_13b_memory, awq_13b_avg, label='AWQ', s=s, color='brown', marker='*', zorder=11)
    ax1[1].scatter(gptq_13b_memory, gptq_13b_avg, label='GPTQ', s=s, color='blue', marker='*', zorder=11)
# plt.xlabel('Bit')
# plt.ylabel('Zero-shot Average')

func0()
# func1()

ax1[0].set_ylim(58, 74)
ax1[0].set_yticks(range(58, 75, 4))
ax1[0].set_xlim(pb_llm_13b_memory[0] - 0.1, 7.2)

ax1[0].axvline(x=awq_13b_memory[1], linestyle='--', linewidth=2, color='black', alpha=0.5)
ax1[0].axvline(x=awq_13b_memory[2], linestyle='--', linewidth=2, color='black', alpha=0.5)

# yticks, xticks 크기 늘리기
ax1[0].tick_params(axis='both', which='major', labelsize=20)
# ax1[1].tick_params(axis='both', which='major', labelsize=20)


ax1[0].legend(['AMQ(Ours)', 'BitStack', 'PB-LLM', 'AWQ', 'GPTQ'], loc='lower right', ncol=2, fontsize=18)
ax1[0].grid()

plt.savefig('source_1_memory_half.png', dpi=500)

