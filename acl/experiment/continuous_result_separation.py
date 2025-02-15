import numpy as np
from matplotlib import pyplot as plt

# owq_llama_2_gemv_hf

# 표로 그릴 것.

colors = [
    '#FF6663',
    '#939393',
    '#C83C04',
    '#378375',
    '#6699FF',
    '#FACA00',
    '#2351AB',
]

# fig_path='acl/experiment/continous_result.png'
fig_path = '/NAS/SJ/nsgaquant/acl/experiment/continuous_result_separation.png'

font = {'size'   : 13}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

linewidth = 2
markersize = 4

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 10)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

our_7b_bits = [2.20369171, 2.40349741, 2.60427461, 2.80246114, 2.99740933, 3.20498705, 3.40123057, 3.60103627, 3.80310881, 3.99222798]
our_7b_w2 = [9.70293808, 8.307990074, 7.460319519, 6.713390827, 6.187410831, 6.020772934, 5.869136333, 5.793658257, 5.68144989, 5.597347736]
our_7b_c4 = [12.82921791, 10.93458462, 9.837593079, 8.866043091, 8.225736618, 8.009811401, 7.793493271, 7.64744854, 7.530736923, 7.44137764]
our_7b_acc = np.array([0.604282098, 0.625704665, 0.637907708, 0.666701336, 0.677821555, 0.682099624, 0.686616407, 0.686102525, 0.688649375, 0.697427872])

bitstack_7b_bits = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
bitstack_7b_w2 = [9.175833702, 8.297012329, 7.791053772, 7.414270401, 7.098052025, 6.766305923, 6.555026531, 6.430410385, 6.334755898, 6.266643047]
bitstack_7b_c4 = [12.56320381, 11.26459026, 10.59273624, 10.04853344, 9.664886475, 9.113240242, 8.809690475, 8.641955376, 8.511131287, 8.385821342]
bitstack_7b_acc = np.array([0.601391018, 0.619275265, 0.627521764, 0.637491537, 0.643394393, 0.656135069, 0.662427668, 0.667206655, 0.672052206, 0.674013289])

pbllm_7b_bits = [2.197, 2.4, 2.603, 2.799, 3.002, 3.198, 3.401, 3.597, 3.8, 4.003]
pbllm_7b_w2 = [25.409021, 17.224394, 13.824484, 11.35747, 9.181018, 8.061, 7.414878, 6.925223, 6.602927, 6.332662]
pbllm_7b_c4 = [32.068573, 22.882645, 17.609436, 14.669633, 12.173552, 10.701463, 9.805017, 9.174158, 8.767661, 8.437231]
pbllm_7b_acc = np.array([0.439466138, 0.480417192, 0.536299553, 0.578924823, 0.613636868, 0.634068865, 0.646246926, 0.659146108, 0.673270014, 0.678828369])

owq_7b_bits = [2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8]
owq_7b_w2 = [9.512156487, 9.513555527, 9.25275135, 9.089015007, 6.015853405, 5.974658489, 5.961352825, 5.967914104]
owq_7b_c4 = [11.93916416, 11.70453358, 11.55443287, 11.22610092, 7.941548347, 7.913356781, 7.89141798, 7.887867451]
owq_7b_acc = np.array([0.60127132, 0.609347297, 0.611218543, 0.616280351, 0.690211332, 0.689057585, 0.687197822, 0.688208984])

ax[0][0].plot(pbllm_7b_bits, pbllm_7b_c4, label='PB-LLM', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
ax[0][0].plot(bitstack_7b_bits, bitstack_7b_c4, label='BitStack', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[0][0].plot(our_7b_bits, our_7b_c4, label='AutoLLMQuant', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[0][0].grid(c='0.8')

ax[1][0].plot(pbllm_7b_bits, 100 * pbllm_7b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
ax[1][0].plot(bitstack_7b_bits, 100 * bitstack_7b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[1][0].plot(our_7b_bits, 100 * our_7b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[1][0].grid(c='0.8')

ax[0][0].tick_params(axis='both', which='major', labelsize=15)
ax[1][0].tick_params(axis='both', which='major', labelsize=15)

pbllm_13b_bits = [2.197, 2.4, 2.603, 2.799, 3.002, 3.198, 3.401, 3.597, 3.8, 4.003]
pbllm_13b_w2 = [49.662487, 21.767782, 12.027323, 9.540066, 7.770907, 6.957156, 6.444432, 6.03354, 5.794348, 5.621876]
pbllm_13b_c4 = [53.145523, 28.799479, 16.243132, 13.139219, 10.783956, 9.700974, 8.908892, 8.274727, 7.935522, 7.717345]
pbllm_13b_acc = np.array([0.445306741, 0.521918144, 0.608104881, 0.64488449, 0.671657683, 0.683852428, 0.695484065, 0.708052548, 0.717220212, 0.713828894])

owq_13b_bits = [2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8]
owq_13b_w2 = [8.141159058, 7.90238905, 7.906852245, 7.644672871, 5.259451866, 5.233787537, 5.246657372, 5.222622395]
owq_13b_c4 = [10.81324387, 10.55164051, 10.35543156, 10.24857807, 7.184574127, 7.174242496, 7.165292263, 7.144797802]
owq_13b_acc = np.array([0.649320442, 0.64854396, 0.652410135, 0.660048735, 0.718132311, 0.716591034, 0.724292107, 0.719481134])

bitstack_13b_bits = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
bitstack_13b_w2 = [9.175833702, 8.297012329, 7.791053772, 7.414270401, 7.098052025, 6.766305923, 6.555026531, 6.430410385, 6.334755898, 6.266643047]
bitstack_13b_c4 = [10.31787777, 9.537553787, 9.025455475, 8.665048599, 8.353291512, 7.974319458, 7.801470757, 7.675413132, 7.572425842, 7.49377346]
bitstack_13b_acc = np.array([0.62639496, 0.65044807, 0.666231766, 0.676121231, 0.682551099, 0.688576983, 0.695072416, 0.696956562, 0.70252, 0.700891267])

our_13b_bits = [2.20309917, 2.40061983, 2.60392562, 2.80206612, 3.00165289, 3.20392562, 3.40123967, 3.60392562, 3.80041322, 3.98615702]
our_13b_w2 = [7.209764004, 6.441867828, 5.988779068, 5.634430885, 5.374492168, 5.24823761, 5.157777786, 5.090263367, 5.024534225, 4.965404987]
our_13b_c4 = [9.835852623, 8.829443932, 8.194261551, 7.670791626, 7.335062981, 7.170888901, 7.054496765, 6.962110519, 6.886018276, 6.837729931]
our_13b_acc = np.array([0.66219152, 0.682863029, 0.691083918, 0.710422927, 0.715917286, 0.71687743, 0.725760049, 0.722983757 ,0.727874139, 0.728116507])


ax[0][1].plot(our_13b_bits, our_13b_c4, label='AutoLLMQuant', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[0][1].plot(pbllm_13b_bits, pbllm_13b_c4, label='PB-LLM', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
ax[0][1].plot(bitstack_13b_bits, bitstack_13b_c4, label='BitStack', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[0][1].grid(c='0.8')

# ax = axes[1].twinx()

ax[1][1].plot(our_13b_bits, 100 * our_13b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[1][1].plot(pbllm_13b_bits, 100 * pbllm_13b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
ax[1][1].plot(bitstack_13b_bits, 100 * bitstack_13b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[1][1].grid(c='0.8')

ax[0][1].tick_params(axis='both', which='major', labelsize=15)
ax[1][1].tick_params(axis='both', which='major', labelsize=15)

bitstack_70b_bits = [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
bitstack_70b_w2 = [4.974086285, 4.728073597,  4.493756294, 4.278894424, 4.069427013, 3.967707634, 3.888819456, 3.816739559, 3.747808456, 3.707520008]
bitstack_70b_c4 = [7.515676975, 7.180526733, 6.885457039, 6.638876915, 6.431142807, 6.316576004, 6.232776165, 6.160410404, 6.090384483, 6.057357311]
bitstack_70b_acc = [0.728957305, 0.733191138, 0.74534594, 0.756955741, 0.764177957, 0.764116636, 0.765076091, 0.767906965, 0.769242697, 0.771363724]
bitstack_70b_acc = np.array(bitstack_70b_acc)

our_70b_bits = [2.20306373, 2.40490196, 2.60294118, 3.00343137, 3.20294118, 3.40490196, 3.60465686, 3.987377451]
our_70b_w2 = [4.993602276, 4.576478481, 4.275177956, 3.724072933, 3.627696037, 3.550760746, 3.496205807, 3.410032511]
our_70b_c4 = [7.389309883, 6.886165619, 6.535670757, 6.018145561, 5.942317963, 5.87950182, 5.835152149, 5.773358345]
our_70b_acc = [0.738021299, 0.743657972, 0.755443093, 0.766131804, 0.769627149, 0.768681573, 0.772659542, 0.776231165]
our_70b_acc = np.array(our_70b_acc)

ax[0][2].plot(our_70b_bits, our_70b_c4, label='AutoLLMQuant', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[0][2].plot(bitstack_70b_bits, bitstack_70b_c4, label='BitStack', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[0][2].grid(c='0.8')

ax[1][2].plot(our_70b_bits, 100 * our_70b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')
ax[1][2].plot(bitstack_70b_bits, 100 * bitstack_70b_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
ax[1][2].grid(c='0.8')

fig.tight_layout() 
plt.savefig(fig_path, dpi=300)
