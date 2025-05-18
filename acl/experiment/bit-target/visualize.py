import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
fig_path = '/NAS/SJ/nsgaquant/acl/experiment/bit-target/result.png'

gptq_7b_memory     = np.array([2238,   2817,  3010,  3589]) / 1024
gptq_7b_wikitext2  = np.array([61.77,   9.27,  6.45,  6.09])
gptq_7b_c4         = np.array([44.10,  11.81,  8.53,  7.86])
gptq_7b_avg        = np.array([43.19,  60.70, 67.22, 68.55])

gptq_13b_memory     = np.array([4029,  5164,  5542,  6676]) / 1024
gptq_13b_wikitext2  = np.array([27.78,  6.75,  5.48,  5.19])
gptq_13b_c4         = np.array([23.39,  8.96,  7.49,  7.06])
gptq_13b_avg        = np.array([45.64, 67.39, 70.89, 71.85])

gptq_70b_memory     = np.array([19363, 25483, 27523, 33643]) / 1024
gptq_70b_wikitext2  = np.array([ 8.33,  4.88,  3.88,  3.59])
gptq_70b_c4         = np.array([10.71,  7.11,  6.11,  5.90])
gptq_70b_avg        = np.array([60.40, 73.31, 76.64, 77.07])

awq_7b_memory       = np.array([2238,   2817,  3010,  3589]) / 1024
awq_7b_wikitext2    = np.array([2.22e5, 15.12,  6.25,  5.83])
awq_7b_c4           = np.array([1.68e5, 17.44,  8.30,  7.72])
awq_7b_avg          = np.array([36.12,  54.67, 67.63, 69.10])

awq_13b_memory      = np.array([4029,  5164,  5542,  6676]) / 1024
awq_13b_wikitext2   = np.array([1.22e5,  6.45,  5.32,  5.06])
awq_13b_c4          = np.array([9.55e4,  9.07,  7.31,  6.96])
awq_13b_avg         = np.array([40.59, 67.34, 72.11, 72.59])

awq_70b_memory      = np.array([19363, 25483, 27523, 33643]) / 1024
awq_70b_wikitext2   = np.array([7.25e4,  4.36,  3.74,  3.48])
awq_70b_c4          = np.array([6.56e4,  6.63,  6.04,  5.84])
awq_70b_avg         = np.array([40.54, 75.10, 76.58, 77.41])

# AMQ
amq_7b_memory       = np.array([2315,  2817,  3010,  3589]) / 1024
amq_7b_wikitext2    = np.array([11.09,  6.85,  6.19,  5.71])
amq_7b_c4           = np.array([14.62,  9.07,  8.23,  7.56])
amq_7b_avg          = np.array([58.00, 66.23, 67.78, 69.05])

amq_13b_memory      = np.array([4181,  5164,  5542,  6676]) / 1024
amq_13b_wikitext2   = np.array([ 7.71,  5.74,  5.37,  5.03])
amq_13b_c4          = np.array([10.43,  7.80,  7.34,  6.90])
amq_13b_avg         = np.array([64.19, 70.37, 71.59, 72.87])

amq_70b_memory      = np.array([20179, 25483, 27523, 33643]) / 1024
amq_70b_wikitext2   = np.array([ 5.18,  4.03,  3.72,  3.46])
amq_70b_c4          = np.array([ 7.63,  6.30,  6.02,  5.81])
amq_70b_avg         = np.array([73.77, 76.28, 76.61, 77.24])


font = {'size'   : 13}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

linewidth = 2
markersize = 6

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
axes.plot(gptq_7b_memory, gptq_7b_c4, label='GPTQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
axes.plot(awq_7b_memory, awq_7b_c4, label='AWQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
axes.plot(amq_7b_memory, amq_7b_c4, label='AMQ', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

axes.plot(gptq_13b_memory, gptq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
axes.plot(awq_13b_memory, awq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
axes.plot(amq_13b_memory, amq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes.plot(gptq_70b_memory, gptq_70b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes.plot(awq_70b_memory, awq_70b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes.plot(amq_70b_memory, amq_70b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

axes.set_ylim(0, 20)
axes.grid()

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5)) 
# fig.subplots_adjust(hspace=0.5, wspace=0.1)

# axes[0].plot(gptq_7b_memory, gptq_7b_wikitext2, label='GPTQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[0].plot(awq_7b_memory, awq_7b_wikitext2, label='AWQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[0].plot(amq_7b_memory, amq_7b_wikitext2, label='AMQ', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[0].plot(gptq_13b_memory, gptq_13b_wikitext2, marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[0].plot(awq_13b_memory, awq_13b_wikitext2, marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[0].plot(amq_13b_memory, amq_13b_wikitext2, marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[0].set_ylim(3, 20)
# axes[0].grid()

# axes[1].plot(gptq_7b_memory, gptq_7b_c4, label='GPTQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[1].plot(awq_7b_memory, awq_7b_c4, label='AWQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[1].plot(amq_7b_memory, amq_7b_c4, label='AMQ', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[1].plot(gptq_13b_memory, gptq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[1].plot(awq_13b_memory, awq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[1].plot(amq_13b_memory, amq_13b_c4, marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[1].set_ylim(3, 20)
# axes[1].grid()

# axes[2].plot(gptq_7b_memory, gptq_7b_avg, label='GPTQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[2].plot(awq_7b_memory, awq_7b_avg, label='AWQ', marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[2].plot(amq_7b_memory, amq_7b_avg, label='AMQ', marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[2].plot(gptq_13b_memory, gptq_13b_avg, marker='o', markersize=markersize, linewidth=linewidth, c = colors[0])
# axes[2].plot(awq_13b_memory, awq_13b_avg, marker='o', markersize=markersize, linewidth=linewidth, c = colors[1])
# axes[2].plot(amq_13b_memory, amq_13b_avg, marker='o', markersize=markersize, linewidth=linewidth, c = 'red')

# axes[2].set_ylim(30, 80)
# axes[2].grid()

fig.tight_layout() 
plt.savefig(fig_path, dpi=300)
