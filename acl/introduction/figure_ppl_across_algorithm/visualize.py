import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# wikitext	c4		piqa	winogrande	hellaswag	arc-c	arc-e	avg
llama_7b = [5.472088814, 7.263512135, 0.791077258, 0.692186267, 0.760007967, 0.462457338, 0.745791246, 0.690304015]
llama_13b = [4.883709431, 6.727157593, 0.805223069, 0.72296764, 0.7936666, 0.491467577, 0.775252525, 0.717715482]

llama_7b_awq_asym_2bit = [15.16874886, 18.61144257, 0.68117519, 0.581689029, 0.546803426, 0.310580205, 0.511784512, 0.526406472]
llama_7b_awq_sym_2bit = [229121.7031, 173262.1563, 0.507072905, 0.487766377, 0.260306712, 0.266211604, 0.259680135, 0.356207547]
llama_7b_gptq_2bit = [61.76620483, 44.10067749, 0.583242655, 0.514601421, 0.401712806, 0.251706485, 0.353956229, 0.421043919]

llama_13b_awq_asym_2bit = [9.053100586, 11.94542789, 0.741022851, 0.63851618, 0.62756423, 0.359215017, 0.623737374, 0.59801113]
llama_13b_awq_sym_2bit = [124759.0313, 97126.03125, 0.5, 0.494869771, 0.260306712, 0.277303754, 0.269781145, 0.360452276]
llama_13b_gptq_2bit = [27.7838192, 23.3856926, 0.62078346, 0.521704815, 0.423720374, 0.256825939, 0.405723906, 0.445751699]

font = {'size'   : 20}
matplotlib.rc('font', **font)
plt.rc('axes', axisbelow=True)

height = 0.5
colors = [
    # '#939393',
    '#2351AB',
    '#C83C04',
    '#FACA00',
    '#378375',
    # '#6699FF'
]


f = plt.figure(figsize=(16, 7))

(ax1, ax2) = f.subplots(ncols=2, nrows=2, sharey=True)

llama_7b_methods = list(['Awq(sym)', 'Awq(asym)', 'GPTQ'])
llama_13b_methods = list(['Awq(sym)', 'Awq(asym)', 'GPTQ'])

height = 0.5
ax1[0].barh(llama_7b_methods, [llama_7b_awq_asym_2bit[0], llama_7b_awq_sym_2bit[0], llama_7b_gptq_2bit[0]], height=height, label='wikitext', color=colors)
ax1[1].barh(llama_7b_methods, [llama_7b_awq_asym_2bit[0], llama_7b_awq_sym_2bit[0], llama_7b_gptq_2bit[0]], height=height, label='wikitext', color=colors)
ax1[0].set_xlim(0, 100)
ax1[1].set_xlim(80000, 250000)

ax2[0].barh(llama_13b_methods, [llama_13b_awq_asym_2bit[0], llama_13b_awq_sym_2bit[0], llama_13b_gptq_2bit[0]], height=height, label='wikitext', color=colors)
ax2[1].barh(llama_13b_methods, [llama_13b_awq_asym_2bit[0], llama_13b_awq_sym_2bit[0], llama_13b_gptq_2bit[0]], height=height, label='wikitext', color=colors)
ax2[0].set_xlim(0, 100)
ax2[1].set_xlim(80000, 250000)

ax1[1].set_xticks(np.arange(100000, 250001, 50000))
ax2[1].set_xticks(np.arange(100000, 250001, 50000))

# f.text(0.5, 0, 'Wikitext2 ppl', ha='center', fontsize=15)

ax1[0].axvline(llama_7b[0], c='red', ls = '--', linewidth=4)
ax2[0].axvline(llama_13b[0], c='red', ls = '--', linewidth=4)
ax1[0].text(7, 0.43, 'Llama-2-7b-hf', color = 'red', fontdict={'size'   : 20, 'fontweight':'bold'})
ax2[0].text(7, 0.43, 'Llama-2-13b-hf', color = 'red', fontdict={'size'   : 20, 'fontweight':'bold'})

plt.tight_layout()
ax1[0].grid()
ax1[1].grid()
ax2[0].grid()
ax2[1].grid()

plt.savefig('Figure_1.png')