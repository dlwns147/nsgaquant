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
fig_path = '/NAS/SJ/nsgaquant/acl/analysis/outlier/outlier'

font = {'size'   : 13}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

linewidth = 3
markersize = 5

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

outlier_bits = [2.19632353, 2.40379902, 2.59865196, 2.7995098, 3.00232843, 3.20318627, 3.40208333, 3.5997549]
outlier_w2 = [5.063642502, 4.755596638, 4.367684364, 4.075111866, 3.827835321, 3.711799622, 3.60807991, 3.538823605]
outlier_c4 = [7.475886822, 7.048167229, 6.642660141, 6.321521759, 6.096700668, 6.005050659, 5.924552441, 5.881522655]
outlier_acc = [0.736838531, 0.744658553, 0.754150306, 0.760177014, 0.764130975, 0.766774821, 0.770105901, 0.773823544]

no_outlier_bits = [2.20306373, 2.40490196, 2.60294118, 2.80416667, 3.00343137, 3.20294118 ,3.40490196, 3.60465686, 3.79730392]
no_outlier_w2 = [4.993602276, 4.576478481, 4.275177956, 3.963034868, 3.724072933, 3.627696037, 3.550760746, 3.496205807, 3.445744038]
no_outlier_c4 = [7.389309883, 6.886165619, 6.535670757, 6.232421398, 6.018145561, 5.942317963, 5.87950182, 5.835152149, 5.799371719]
no_outlier_acc = [0.738021299, 0.743657972, 0.755443093, 0.76499475, 0.766131804, 0.769627149, 0.768681573, 0.772659542, 0.773729407]


axes.plot(outlier_bits, outlier_c4, label='w/ outliers', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
axes.plot(no_outlier_bits, no_outlier_c4, label='w/o outliers', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])
axes.grid(c='0.8')

# ax = axes.twinx()
# ax.plot(outlier_bits, outlier_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[2])
# ax.plot(no_outlier_bits, no_outlier_acc, linestyle = '--', marker='o', markersize=markersize, linewidth=linewidth, c = colors[5])

# ax.grid(c='0.8')
# ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
# ax.tick_params(axis='both', which='major', labelsize=15)
axes.tick_params(axis='both', which='major', labelsize=15)


fig.tight_layout() 
plt.savefig(f'{fig_path}.png', dpi=300)
plt.savefig(f'{fig_path}.pdf', dpi=300)
