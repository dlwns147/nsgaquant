import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


linear_7b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_linear/greedy_Llama-2-7b-hf_wikitext2_ppl.json'
module_7b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_module/greedy_Llama-2-7b-hf_wikitext2_ppl.json'
layer_7b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_layer/greedy_Llama-2-7b-hf_wikitext2_ppl.json'

linear_13b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_linear/greedy_Llama-2-13b-hf_wikitext2_ppl.json'
module_13b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_module/greedy_Llama-2-13b-hf_wikitext2_ppl.json'
layer_13b = '/NAS/SJ/nsgaquant/acl/analysis/figure_5/greedy_hqq_descending_42_layer/greedy_Llama-2-13b-hf_wikitext2_ppl.json'

with open(linear_7b, 'r') as f:
    linear_7b = json.load(f)['archive']

with open(module_7b, 'r') as f:
    module_7b = json.load(f)['archive']

with open(layer_7b, 'r') as f:
    layer_7b = json.load(f)['archive']

with open(linear_13b, 'r') as f:
    linear_13b = json.load(f)['archive']

with open(module_13b, 'r') as f:
    module_13b = json.load(f)['archive']

with open(layer_13b, 'r') as f:
    layer_13b = json.load(f)['archive']

# font = {'size'   : 20}
# matplotlib.rc('font', **font)
# plt.rc('axes', axisbelow=True)

# height = 0.5
colors = [
    '#939393',
    '#2351AB',
    '#C83C04',
    '#FACA00',
    '#378375',
    '#6699FF'
]

linear_7b_ppl = [entry['ppl'] for entry in linear_7b if 'ppl' in entry][1:]
linear_7b_bit = [entry['bit'] for entry in linear_7b if 'ppl' in entry][1:]

module_7b_ppl = [entry['ppl'] for entry in module_7b if 'ppl' in entry][1::2]# + [module_7b[0]['ppl']]
module_7b_bit = [entry['bit'] for entry in module_7b if 'ppl' in entry][1::2]# + [4]

layer_7b_ppl = [entry['ppl'] for entry in layer_7b if 'ppl' in entry][1::2]# + [module_7b[0]['ppl']]
layer_7b_bit = [entry['bit'] for entry in layer_7b if 'ppl' in entry][1::2]# + [4]

linear_13b_ppl = [entry['ppl'] for entry in linear_13b if 'ppl' in entry]
linear_13b_bit = [entry['bit'] for entry in linear_13b if 'ppl' in entry]

module_13b_ppl = [entry['ppl'] for entry in module_13b if 'ppl' in entry][::2]# + [module_7b[0]['ppl']]
module_13b_bit = [entry['bit'] for entry in module_13b if 'ppl' in entry][::2]# + [4]

layer_13b_ppl = [entry['ppl'] for entry in layer_13b if 'ppl' in entry][1::2]# + [module_7b[0]['ppl']]
layer_13b_bit = [entry['bit'] for entry in layer_13b if 'ppl' in entry][1::2]# + [4]

fig, ax = plt.subplots(1,2, figsize=(16, 7))

# ax[0].set_yscale('log')
ax[0].set_ylim(5, 20)
# ax[1].set_yscale('log')
ax[1].set_ylim(5, 20)

x = np.arange(2.2, 4, 0.2)
x = list(x)

print(linear_13b_bit)
print(module_13b_bit)
print(layer_13b_bit)
print(linear_13b_ppl)
print(module_13b_ppl)
print(layer_13b_ppl)

# ax[0].plot(linear_7b_bit, linear_7b_ppl, label='Linear', color = colors[4])
# ax[0].plot(module_7b_bit, module_7b_ppl, label='Module', color = colors[2])
# ax[0].plot(layer_7b_bit, layer_7b_ppl, label='Layer', color = colors[1])


ax[0].bar(x, layer_7b_ppl[1:-1][::-1], label='Layer', color = colors[1], width = 0.1)
ax[0].bar(x, module_7b_ppl[1:-1][::-1], label='Module', color = colors[3], width = 0.1)
ax[0].bar(x, linear_7b_ppl[1:-1][::-1], label='Linear', color = colors[4], width = 0.1)
ax[0].legend()

ax[1].bar(x, layer_13b_ppl[1:][::-1], label='Layer', color = colors[1], width = 0.1)
ax[1].bar(x, module_13b_ppl[1:][::-1], label='Module', color = colors[3], width = 0.1)
ax[1].bar(x, linear_13b_ppl[1:][::-1], label='Linear', color = colors[4], width = 0.1)
ax[1].legend()

# plt.xlabel('Bit')
# plt.ylabel('Wikitext2 ppl')

plt.savefig('figure_5.png')