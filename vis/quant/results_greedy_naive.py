import csv
import json
import numpy as np
import matplotlib.pyplot as plt
# from utils import get_net_info

ppl_arch_figure = '/NAS/SJ/nsgaquant/vis/fig/quant/results_greedy_naive.png'

# model_name='Llama-2-7b-hf'

# config='/NAS/SJ/nsgaquant/config/llama.json'
# with open(config, 'r') as f:
#     config = json.load(f)[model_name]

greedy_2_7b_bits = [2.194300518, 2.399611399, 2.599740933, 2.799870466, 3.001619171, 3.203367876, 3.399611399, 3.600712435, 3.801165803]
greedy_2_7b_c4 = [22.64138222, 16.08280945, 13.56331921, 11.9564352, 10.72120094, 9.680900574, 8.760696411, 8.147957802, 7.771416664]
# greedy_2_7b_acc = [44.8172871, 50.1960435, 53.3706148, 55.7299494, 57.5801991, 59.1751392, 61.380781, 63.0346308, 63.8663452]
greedy_2_7b_acc = [46.8705445, 51.3036559, 53.7252329, 55.6420499, 57.2068914, 58.4699551, 60.6577838, 62.2175692, 62.87671]

naive_2_7b_bits = [2.194300518, 2.394430052, 2.599740933, 2.794689119, 3.003562176, 3.205310881, 3.399935233, 3.602331606, 3.801165803]
naive_2_7b_c4 = [34.92710876, 18.41146278, 14.43975639, 13.09913921, 11.1363802, 9.877692223, 9.044061661, 8.160598755, 7.786623478]

naive_2_7b_acc = [42.8722042, 48.4534727, 51.3939514, 52.1130229, 55.5329162, 57.9993595, 60.3216575, 61.6871489, 62.8939619]


greedy_2_13b_bits = [2.194628099, 2.404958678, 2.598760331, 2.804958678, 2.997107438, 3.196280992, 3.402066116, 3.6, 3.8]
greedy_2_13b_c4 = [14.64899158, 11.55554008, 10.36480236, 9.442017555, 8.704010963, 8.135162354, 7.649227142, 7.230648994, 6.991567135]
# greedy_2_13b_acc = []
greedy_2_13b_acc = [52.857662, 56.5447569, 58.47788, 60.3442165, 61.539811, 63.1647994, 63.8714747, 64.7899821, 65.5156557]

naive_2_13b_bits = [2.195867769, 2.400826446, 2.598760331, 2.799586777, 2.997520661, 3.196694215, 3.398347107, 3.6, 3.801239669]
naive_2_13b_c4 = [18.09353256, 12.28570843, 10.69265652, 9.826218605, 9.033518791, 8.398552895, 7.761595249, 7.256059647, 6.992651939]

naive_2_13b_acc = [47.8661515, 54.0076631, 56.6527274, 57.8121102, 59.5191813, 61.0694543, 63.3483813, 65.0615661, 65.6191532]


greedy_31_8b_bits = [2.199519231, 2.403846154, 2.59375, 2.794471154, 3.001201923, 3.198317308, 3.403846154, 3.594951923, 3.800480769]
greedy_31_8b_c4 = [715.486145, 83.77993774, 43.30444336, 28.9252491, 22.82278633, 18.9542942, 15.92402077, 14.23091221, 12.18838596]

greedy_31_8b_acc = [37.0543551, 43.9461266, 48.2173047, 51.7015383, 55.2770315, 58.6520044, 60.865522, 62.5672113, 65.0371422]

naive_31_8b_bits = [2.200721154, 2.405048077, 2.602163462, 2.801682692, 2.997596154, 3.193509615, 3.393028846, 3.604567308, 3.800480769]
naive_31_8b_c4 = [4064.260498, 1419.790405, 222.3893127, 102.086853, 61.54899216, 34.76722336, 21.05095291, 15.66402054, 12.39433289]

naive_31_8b_acc = [35.2021797, 35.868828, 38.4014908, 41.2193127, 44.3081579, 48.1058975, 55.359733, 60.7198405, 64.6116829]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

color = ['red', 'black']
color = ['green', 'purple']
color = ['darkblue', 'darkred']
color = ['#1f77b4', '#ff7f0e']

font_size=15
legend_font_size=14

axes[0].plot(naive_2_7b_bits, naive_2_7b_c4, 'o-', label='Naive Search C4 PPL', color=color[0])
axes[0].plot(greedy_2_7b_bits, greedy_2_7b_c4, 'o-', label='Greedy Search C4 PPL', color=color[1])
axes[0].set_title('Llama 2 7B', fontsize=font_size)
# axes[0].set_xlabel('Bits')
axes[0].set_ylabel('C4 PPL', fontsize=font_size)
axes[0].grid(c='0.8')
axes[0].set_axisbelow(True)
# axes[0].set_ylim([6, None])
axes[0].set_yticks(list(range(5, 35 + 1, 5)))


axes2 = axes[0].twinx()
axes2.plot(naive_2_7b_bits, naive_2_7b_acc, 'o--', label='Naive Search Avg. Acc.', color=color[0])
axes2.plot(greedy_2_7b_bits, greedy_2_7b_acc, 'o--', label='Greedy Search Avg. Acc.', color=color[1])
# axes2.set_ylabel('Avg. Acc.')
axes2.set_yticks(list(range(40, 65 + 1, 5)))
axes2.grid(c='0.8')
axes2.set_axisbelow(True)


axes[1].plot(naive_2_13b_bits, naive_2_13b_c4, 'o-', label='Greedy Search C4 PPL', color=color[0])
axes[1].plot(greedy_2_13b_bits, greedy_2_13b_c4, 'o-', label='Naive Search C4 PPL', color=color[1])
axes[1].set_title('Llama 2 13B', fontsize=font_size)
# axes[1].set_xlabel('Bits')
# axes[1].set_ylabel('C4 PPL')
axes[1].grid(c='0.8')
axes[1].set_axisbelow(True)
# axes[1].set_ylim([6, None])
# axes[1].set_yticks(list(range(5, 35 + 1, 5)))


axes2 = axes[1].twinx()
axes2.plot(naive_2_13b_bits, naive_2_13b_acc, 'o--', label='Greedy Search Avg. Acc.', color=color[0])
axes2.plot(greedy_2_13b_bits, greedy_2_13b_acc, 'o--', label='Naive Search Avg. Acc.', color=color[1])
# axes2.set_ylabel('Avg. Acc.', fontsize=font_size)
# axes2.set_yticks(list(range(40, 65 + 1, 5)))
axes2.grid(c='0.8')
axes2.set_axisbelow(True)

axes[2].plot(naive_31_8b_bits, naive_31_8b_c4, 'o-', label='Greedy Search C4 PPL', color=color[0])
axes[2].plot(greedy_31_8b_bits, greedy_31_8b_c4, 'o-', label='Naive Search C4 PPL', color=color[1])
axes[2].set_yscale('log', base=10) 
axes[2].set_title('Llama 3.1 8B', fontsize=font_size)
# axes[2].set_xlabel('Bits')
# axes[1].set_ylabel('C4 PPL')
axes[2].grid(c='0.8')
axes[2].set_axisbelow(True)
# axes[0].set_ylim([6, None])
# axes[0].set_yticks(list(range(5, 35 + 1, 5)))


axes2 = axes[2].twinx()
axes2.plot(naive_31_8b_bits, naive_31_8b_acc, 'o--', label='Greedy Search Avg. Acc.', color=color[0])
axes2.plot(greedy_31_8b_bits, greedy_31_8b_acc, 'o--', label='Naive Search Avg. Acc.', color=color[1])
axes2.set_ylabel('Avg. Acc.', fontsize=font_size)
# axes2.set_yticks(list(range(40, 65 + 1, 5)))
axes2.grid(c='0.8')
axes2.set_axisbelow(True)

fig.text(0.5, 0.04, 'Bits', ha='center', fontsize=font_size)
handles, labels = axes[-1].get_legend_handles_labels()
handels_2, labels_2 = axes2.get_legend_handles_labels()
handles = handles + handels_2
labels = labels + labels_2

fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.00), fontsize=legend_font_size)
plt.tight_layout(rect=[0, 0.05, 1, 0.9])

# fig.tight_layout() 
plt.show()
plt.savefig(ppl_arch_figure, dpi=300)