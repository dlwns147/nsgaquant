import numpy as np
from matplotlib import pyplot as plt

# amq_13b_bit = np.array([2.104545455, 2.20309917, 2.25454545, 2.40061983, 2.50433884, 2.60392562, 2.74938017, 2.80206612, 3.00165289, 3.10371901, 3.20392562, 3.25454545, 3.40123967, 3.50061983, 3.60392562, 3.75475207, 3.80041322, 3.98615702]) + 0.25
# amq_13b_avg = np.array([0.641908905, 0.66219152, 0.665796602, 0.682863029, 0.689958569, 0.691083918, 0.703693136, 0.710422927, 0.715917286, 0.718818017, 0.71687743, 0.72231014, 0.725760049, 0.724154011, 0.722983757, 0.728700119, 0.727874139, 0.728116507]) * 100

# owq_13b_bit = np.array([2.2,2.4,2.5,2.6,2.8,3.2,3.4,3.5,3.6,3.8]) + 0.25
# owq_13b_avg = np.array([0.649320442, 0.64854396, 0.651275389, 0.652410135, 0.660048735, 0.718132311, 0.716591034, 0.720321308, 0.724292107, 0.719481134]) * 100

# pb_llm_13b_bit = np.array([2.001, 2.197, 2.253, 2.4, 2.498, 2.603, 2.75, 2.799, 3.002, 3.198, 3.247, 3.401, 3.499, 3.597, 3.8, 4.003]) + 0.25
# pb_llm_13b_avg = np.array([0.411433171, 0.445306741, 0.465469785, 0.521918144, 0.582221528, 0.608104881, 0.634757553, 0.64488449, 0.671657683, 0.683852428, 0.686756827, 0.695484065, 0.706756403, 0.708052548, 0.717220212, 0.713828894]) * 100

# bitstack_13b_bit = np.array([2.1, 2.2, 2.25, 2.4, 2.5, 2.6, 2.75, 2.8, 3, 3.2, 3.25, 3.4, 3.5, 3.6, 3.8, 4]) + 0.25
# bitstack_13b_avg = np.array([0.632752944, 0.646154821, 0.652069036, 0.671499793, 0.681265789, 0.687761946, 0.692470769, 0.693403778, 0.704266588, 0.708421695, 0.708730449, 0.714242304, 0.714063108, 0.715914363, 0.720499593, 0.719499094]) * 100

# # group size = 128, symmetric
# awq_13b_bit = np.array([2, 3, 4]) + 0.25
# awq_13b_avg = np.array([0.393870517, 0.679394636, 0.724932406]) * 100

# gptq_13b_bit = np.array([2, 3, 4]) + 0.25
# gptq_13b_avg = np.array([0.370548899, 0.673860776, 0.718537608]) * 100

amq_13b_bit = '2.104545455  2.20309917  2.25454545  2.40061983  2.50433884  2.60392562  2.74938017  2.80206612  3.00165289  3.10371901  3.20392562  3.25454545  3.40123967  3.50061983  3.60392562  3.75475207  3.80041322  3.98615702'.split()
amq_13b_bit = np.array([float(i) + 0.25 for i in amq_13b_bit])
amq_13b_avg = '0.644078055  0.665760234  0.669986952  0.686223257  0.691528902  0.694711336  0.707182622  0.713616369  0.719462467  0.722780501  0.721754772  0.725773749  0.729812015  0.729267104  0.728153136  0.733967738  0.733453817  0.733522945'.split()
amq_13b_avg = [float(i) * 100 for i in amq_13b_avg]

owq_13b_bit = '2.1  2.2  2.4  2.5  2.6  2.8  3.1  3.2  3.4  3.5  3.6  3.8'.split()
owq_13b_bit = np.array([float(i) + 0.25 for i in owq_13b_bit])
owq_13b_avg = '0.645647907  0.65971804  0.660327747  0.663888788  0.665831736  0.67365437  0.727218834  0.726628449  0.725861817  0.728726517  0.732324121  0.72775686'.split()
owq_13b_avg = [float(i) * 100 for i in owq_13b_avg]

pb_llm_13b_bit = '2.001  2.197  2.253  2.4  2.498  2.603  2.75  2.799  3.002  3.198  3.247  3.401  3.499  3.597  3.8  4.003'.split()
pb_llm_13b_bit = np.array([float(i) + 0.25 for i in pb_llm_13b_bit])
pb_llm_13b_avg = '0.375445393  0.408111608  0.434459671  0.498091685  0.576311309  0.608671418  0.644047408  0.653864286  0.681858342  0.694944675  0.697739114  0.706494866  0.717071733  0.717489636  0.726595174  0.724603192'.split()
pb_llm_13b_avg = [float(i) * 100 for i in pb_llm_13b_avg]

bitstack_13b_bit = '2.1  2.2  2.25  2.4  2.5  2.6  2.75  2.8  3  3.2  3.25  3.4  3.5  3.6  3.8  4'.split()
bitstack_13b_bit = np.array([float(i) + 0.25 for i in bitstack_13b_bit])
bitstack_13b_avg = '0.641746985  0.654953141  0.660521485  0.680336852  0.690565154  0.696604581  0.700197145  0.702050346  0.71319105  0.717778326  0.718098418  0.723127819  0.723112838  0.724450119  0.728019916  0.727827701'.split()
bitstack_13b_avg = [float(i) * 100 for i in bitstack_13b_avg]

# awq_13b_bit = np.array([2, 3, 4]) + 0.25
# awq_13b_avg = '40.58586574  72.1126803  72.89647707'.split()
# awq_13b_avg = [float(i) for i in awq_13b_avg]

# gptq_13b_bit = np.array([2, 3, 4]) + 0.25
# gptq_13b_avg = '45.6424071  70.88767941  72.87781402'.split()
# gptq_13b_avg = [float(i) for i in gptq_13b_avg]

awq_13b_bit = np.array([3, 4])
awq_13b_avg = '67.3409126  72.58648748'.split()
awq_13b_avg = [float(i) for i in awq_13b_avg]

gptq_13b_bit = np.array([3, 4])
gptq_13b_avg = '67.38607757  71.85376084'.split()
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
one_bit_memory = 1512.5
scale_zero_memory = 378.125
embed_head_memory = 625
norm_size = 0.791015625
amq_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in amq_13b_bit]
owq_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in owq_13b_bit]
pb_llm_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in pb_llm_13b_bit]
bitstack_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in bitstack_13b_bit]
awq_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in awq_13b_bit]
gptq_13b_memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in gptq_13b_bit]

linestyle = '--'
markersize = 8
linewidth = 3
s = 150

def func0():
    ax1[0].plot(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', marker='o', markersize=markersize, color='r', linewidth=linewidth, zorder=10)
    # ax1[0].bar(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', color='grey', width = 50, linewidth=linewidth, zorder=1)
    # ax1[0].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=markersize, color=colors[1], linewidth=linewidth, zorder=1)
    ax1[0].plot(bitstack_13b_memory, bitstack_13b_avg, linestyle = linestyle, label='BitStack', marker='o', markersize=markersize, color='orange', linewidth=linewidth, zorder = 3)
    ax1[0].plot(pb_llm_13b_memory, pb_llm_13b_avg, linestyle = linestyle, label='PB-LLM', marker='o', markersize=markersize, color=colors[4], linewidth=linewidth)

    ax1[0].scatter(awq_13b_memory, awq_13b_avg, label='AWQ', s=s, color='blue', marker='*', zorder=11)
    ax1[0].scatter(gptq_13b_memory, gptq_13b_avg, label='GPTQ', s=s, color='brown', marker='*', zorder=11)

def func1():
    ax1[1].plot(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', marker='o', markersize=markersize, color='r', linewidth=linewidth, zorder=10)
    # ax1[1].bar(amq_13b_memory, amq_13b_avg, linestyle = linestyle, label='AMQ', color='grey', width = 50, linewidth=linewidth, zorder=1)
    # ax1[1].plot(owq_13b_memory, owq_13b_avg, label='OWQ', marker='o', markersize=markersize, color=colors[1], linewidth=linewidth, zorder=1)
    ax1[1].plot(pb_llm_13b_memory, pb_llm_13b_avg, linestyle = linestyle, label='PB-LLM', marker='o', markersize=markersize, color=colors[4], linewidth=linewidth)
    ax1[1].plot(bitstack_13b_memory, bitstack_13b_avg, linestyle = linestyle, label='BitStack', marker='o', markersize=markersize, color='orange', linewidth=linewidth, zorder=3)
    
    ax1[1].scatter(awq_13b_memory, awq_13b_avg, label='AWQ', s=s, color='blue', marker='*', zorder=11)
    ax1[1].scatter(gptq_13b_memory, gptq_13b_avg, label='GPTQ', s=s, color='brown', marker='*', zorder=11)
# plt.xlabel('Bit')
# plt.ylabel('Zero-shot Average')

func0()
func1()

ax1[0].set_ylim(62, 74)
ax1[1].set_ylim(36, 49)
ax1[0].set_yticks(np.arange(62, 75, 4))
ax1[1].set_yticks(np.arange(36, 49, 4))

# yticks, xticks 크기 늘리기
ax1[0].tick_params(axis='both', which='major', labelsize=20)
ax1[1].tick_params(axis='both', which='major', labelsize=20)

bit = np.array([2.5])
memory = [(bit - 0.25) * one_bit_memory + scale_zero_memory + embed_head_memory for bit in bit]
fp16_13b_tps = 27.3062883732963
bitstack_13b_fused_2_tps = [7.15578038, 6.273436638, 5.496908988]
bitstack_13b_fused_3_tps = [11.89239396, 10.66521603, 9.800740556]
amq_13b_tps = [70.42509431, 75.60473393, 82.53214564]

ax2 = ax1[0].twinx()
ax2.bar(memory, bitstack_13b_fused_2_tps, width=50, color=colors[3], label='BitStack-Fused-2', edgecolor = 'black', zorder = 1)
ax2.bar(memory, amq_13b_tps, width=50, color=colors[2], label='AMQ', edgecolor = 'black', zorder = 1)



# plt.axvline(24, c='red', ls = '--', linewidth=4)
# ax1[0].axvline(2.642039, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(2.642039, c='black', ls = '--', linewidth=4)
# ax1[0].axvline(3.30319602, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(3.30319602, c='black', ls = '--', linewidth=4)
# ax1[0].axvline(3.96435305, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(3.96435305, c='black', ls = '--', linewidth=4)
# ax1[0].axvline(4029, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(4029, c='black', ls = '--', linewidth=4)
# ax1[0].axvline(5542, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(5542, c='black', ls = '--', linewidth=4)
# ax1[0].axvline(7054, c='black', ls = '--', linewidth=4)
# ax1[1].axvline(7054, c='black', ls = '--', linewidth=4)

#f.title('M')

# plt.text(24.5, 4.3, 'RTX 3090', color = 'red', fontdict={'size'   : 20, 'fontweight':'bold'})
# ax1[0].text(2.55, 74.5, '5GB', color = 'black', fontdict={'size'   : 20, 'fontweight':'bold'})
# ax1[0].text(3.21, 74.5, '6GB', color = 'black', fontdict={'size'   : 20, 'fontweight':'bold'})
# ax1[0].text(3.87, 74.5, '7GB', color = 'black', fontdict={'size'   : 20, 'fontweight':'bold'})

# plt.tight_layout()
# ax1[1].legend(['amq', 'OWQ', 'PB-LLM', 'BitStack', 'AWQ', 'GPTQ'], loc='lower right')
ax1[1].legend(['AMQ(Ours)', 'PB-LLM', 'BitStack', 'AWQ', 'GPTQ'], loc='lower right', ncol=2)
ax1[0].grid()
ax1[1].grid()
plt.savefig('source_1_memory.png', dpi=500)

# font = {'size'   : 20}
# plt.rc('font', **font)
# plt.rc('axes', axisbelow=True)

# f = plt.figure(figsize=(5, 5))
# ax1 = [f.subplots(ncols=1, nrows=1)]

# linestyle = '--'
# markersize = 20
# linewidth = 10
# s = 1000

# func0()

# ax1[0].set_ylim(72, 74)
# ax1[0].set_xlim(4.225, 4.26)
# ax1[0].yaxis.tick_right()
# ax1[0].set_yticks([72, 72.5, 73, 73.5, 74])
# # ax1[0].set_xticks([4.2, 4.25, 4.3])
# ax1[0].grid()
# ax1[0].set_frame_on(False)
# f.subplots_adjust(right=0.85)
# plt.savefig('source_2.png', dpi=500)

# # f = plt.figure(figsize=(5, 7))
# # ax1 = [f.subplots(ncols=1, nrows=1)]

# # linestyle = '--'
# # markersize = 10
# # linewidth = 5
# # s = 1000

# # func0()

# # ax1[0].set_ylim(35, 70)
# # ax1[0].set_xlim(2.225, 2.8)
# # ax1[0].yaxis.tick_right()
# # ax1[0].set_yticks([35, 40, 45, 50, 55, 60, 65, 70])
# # # ax1[0].set_xticks([4.2, 4.25, 4.3])
# # ax1[0].grid()
# # ax1[0].set_frame_on(False)
# # f.subplots_adjust(right=0.85)
# # plt.savefig('source_3.png', dpi=500)

# font = {'size'   : 20}
# plt.rc('font', **font)

# f = plt.figure(figsize=(5, 7))
# # ax1, ax2 = f.subplots(ncols=2, nrows=1)
# ax1 = f.subplots(ncols=1, nrows=1)

# # w2
# # ax1.bar([1], [awq_13b_w2], color='b', width=0.4, label='AWQ')
# # ax1.bar([2], [gptq_13b_w2], color='brown', width=0.4, label='GPTQ')
# # ax1.bar([3], [pb_llm_13b_w2], color=colors[4], width=0.4, label='PB-LLM')
# # ax1.bar([4], [bitstack_13b_w2], color='orange', width=0.4, label='BitStack')

# # ax1.axhline(amq_13b_w2, color='r', linestyle='--', linewidth=2)

# # c4
# # ax1.bar([1], [awq_13b_c4], color='b', width=0.4, label='AWQ')
# # ax1.bar([2], [gptq_13b_c4], color='brown', width=0.4, label='GPTQ')
# # ax1.bar([3], [pb_llm_13b_c4], color=colors[4], width=0.4, label='PB-LLM')
# # ax1.bar([4], [bitstack_13b_c4], color='orange', width=0.4, label='BitStack')
# ax1.bar([1], [awq_13b_c4], color='b', width=0.4)
# ax1.bar([2], [gptq_13b_c4], color='brown', width=0.4)
# ax1.bar([3], [pb_llm_13b_c4], color=colors[4], width=0.4)
# ax1.bar([4], [bitstack_13b_c4], color='orange', width=0.4)

# ax1.axhline(amq_13b_c4, color='r', linestyle='--', linewidth=2, label='AMQ(Ours)')

# # ax1.set_ylim(0, 30)
# ax1.set_ylim(5, 25)
# ax1.set_yticks(np.arange(5, 26, 5))

# ax1.legend(fontsize=20, loc='upper right')
# # ax1.text(2.7, 11, 'AMQ(Ours)', color = 'r', fontdict={'size'   : 15, 'fontweight':'bold'})
# ax1.set_xticklabels([None, 'AWQ', 'GPTQ', 'PB-LLM', 'BitStack'], rotation=45)
# # ax1.set_xticks([0, 2.25, 2.25, 2.35, 2.35])

# ax1.grid()
# # ax1.set_frame_on(False)
# f.subplots_adjust(bottom = 0.2)

# plt.savefig('source_4.png', dpi=500)