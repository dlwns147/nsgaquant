import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.decomposition.asf import ASF
import numpy as np
import csv
import numpy.ma as ma

fig_path = f'fig/layer/selected_layer.png'

our_layer_7b_path = '/NAS/SJ/nsgaquant/save/search/2412090938_Llama-2-7b-hf_sparsity_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_128.stats'
our_layer_13b_path = '/NAS/SJ/nsgaquant/save/search/2412091008_Llama-2-13b-hf_sparsity_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.4_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_160.stats'

# greedy_path = '/NAS/SJ/sleb/csv/new/Llama-2-7b-hf_loss_128_js.csv'
# greedy_last_layer_60 = '22.self_attn'
# greedy_last_layer_80 = '18.self_attn'

# with open(greedy_path, 'r') as f:
#     selected_layers = list(csv.reader(f))[0]
# greedy_7b_60_attn = [1] * 32
# greedy_7b_80_attn = [1] * 32
# greedy_7b_60_mlp = [1] * 32
# greedy_7b_80_mlp = [1] * 32

# for layer in selected_layers:
#     blk, layer = layer.split('.')
#     blk = int(blk)
#     if layer == 'self_attn':
#         greedy_7b_80_attn[blk] = 0
#     elif layer == 'mlp':
#         greedy_7b_80_mlp[blk] = 0
#     else:
#         raise NotImplementedError
    
#     if f'{blk}.{layer}' == greedy_last_layer_80:
#         break

# for layer in selected_layers:
#     blk, layer = layer.split('.')
#     blk = int(blk)
#     if layer == 'self_attn':
#         greedy_7b_60_attn[blk] = 0
#     elif layer == 'mlp':
#         greedy_7b_60_mlp[blk] = 0
#     else:
#         raise NotImplementedError
    
#     if f'{blk}.{layer}' == greedy_last_layer_60:
#         break

our_layer_7b_60_arch = {'layer': {'self_attn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], 'mlp': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]}} # 0.6
our_layer_7b_80_arch = {'layer': {'self_attn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1], 'mlp': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]}} # 0.8
our_layer_13b_60_arch = {'layer': {'self_attn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 'mlp': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]}} # 0.6
our_layer_13b_80_arch = {'layer': {'self_attn': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 'mlp': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]}} # 0.8

our_layer_7b_60_attn = our_layer_7b_60_arch['layer']['self_attn']
our_layer_7b_60_mlp = our_layer_7b_60_arch['layer']['mlp']
our_layer_7b_80_attn = our_layer_7b_80_arch['layer']['self_attn']
our_layer_7b_80_mlp = our_layer_7b_80_arch['layer']['mlp']

layer_name = ['Attn', 'Mlp']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 3))
fig.subplots_adjust(hspace=-0.7, wspace=0.1)

# greedy_7b_80 = np.stack([greedy_7b_80_attn, greedy_7b_80_mlp])
# masked_greedy_7b_80 = ma.masked_array(greedy_7b_80, mask=greedy_7b_80)

# axes[0, 0].matshow(masked_greedy_7b_80)
# axes[0, 0].set_yticklabels(['']+layer_name)
# # axes[0, 0].set_xlabel('Layer Index')
# axes[0, 0].set_title('FinerCut')

# greedy_7b_60 = np.stack([greedy_7b_60_attn, greedy_7b_60_mlp])
# masked_greedy_7b_60 = ma.masked_array(greedy_7b_60, mask=greedy_7b_60)

# axes[1, 0].matshow(masked_greedy_7b_60)
# axes[1, 0].set_yticklabels(['']+layer_name)
# axes[1, 0].set_xlabel('Layer Index')

our_layer_7b_80 = np.stack([our_layer_7b_80_arch['layer']['self_attn'], our_layer_7b_80_arch['layer']['mlp']])
masked_our_layer_7b_80 = ma.masked_array(our_layer_7b_80, mask=our_layer_7b_80)

axes[0, 0].set_title("Llama 2 7B")
axes[0, 0].matshow(masked_our_layer_7b_80)
axes[0, 0].set_yticklabels(['']+layer_name)
# axes[0, 0].set_ylabel("20%", rotation=0)
# axes[0, 0].set_title('Our Layer')

our_layer_7b_60 = np.stack([our_layer_7b_60_arch['layer']['self_attn'], our_layer_7b_60_arch['layer']['mlp']])
masked_our_layer_7b_60 = ma.masked_array(our_layer_7b_60, mask=our_layer_7b_60)

axes[1, 0].matshow(masked_our_layer_7b_60)
axes[1, 0].set_yticklabels(['']+layer_name)
axes[1, 0].set_xticklabels([])
axes[1, 0].set_xlabel('Layer Index')
# axes[1, 0].set_ylabel("40%", rotation=0)
# axes[1, 1].set_title('Our Layer')


our_layer_13b_80 = np.stack([our_layer_13b_80_arch['layer']['self_attn'], our_layer_13b_80_arch['layer']['mlp']])
masked_our_layer_13b_80 = ma.masked_array(our_layer_13b_80, mask=our_layer_13b_80)

axes[0, 1].set_title("Llama 2 13B")
axes[0, 1].matshow(masked_our_layer_13b_80)
# axes[0, 1].set_yticklabels(['']+layer_name)
axes[0, 1].set_yticklabels([])
# axes[0, 1].set_title('Our Layer')

our_layer_13b_60 = np.stack([our_layer_13b_60_arch['layer']['self_attn'], our_layer_13b_60_arch['layer']['mlp']])
masked_our_layer_7b_60 = ma.masked_array(our_layer_13b_60, mask=our_layer_13b_60)

axes[1, 1].matshow(masked_our_layer_7b_60)
# axes[1, 1].set_yticklabels(['']+layer_name)
axes[1, 1].set_yticklabels([])
axes[1, 1].set_xticklabels([])
axes[1, 1].set_xlabel('Layer Index')


plt.show()
plt.savefig(fig_path, dpi=300)