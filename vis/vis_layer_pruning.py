import json
from utils.func import get_net_info
import matplotlib.pyplot as plt

model_name = 'Llama-2-7b-hf'
iter = 200
# arch_path = f'/NAS/SJ/nsgaquant/save/search/2411210941_{model_name}_bits_loss_hqq_layer_prune_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_0.95_1.0_linear_group/iter_{iter}.stats'
# arch_path = f'/NAS/SJ/nsgaquant/save/search/2411210941_{model_name}_bits_loss_hqq_layer_prune_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_0.95_1.0_linear_group/iter_{iter}.stats'
arch_path = f'save/search/2411211115_Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_0.96_1.0/iter_{iter}.stats'
config_path = 'config/llama.json'
fig_path = 'fig/bit_sparsity.png'

with open(config_path, 'r') as f:
    config = json.load(f)[model_name]

with open(arch_path, 'r') as f:
    arch = json.load(f)
archive = [a[0] for a in arch['archive']]
candidates = [a[0] for a in arch['candidates']]

archive_sparsity = list()
archive_bits = list()

# for arch in archive:
#     complexity = get_net_info(arch, config)
#     archive_sparsity.append(complexity['sparsity'])
#     archive_bits.append(complexity['bits'])

# for b, s in zip(archive_bits, archive_sparsity):
#     print(f'b : {b:.2f}, s : {s:.2f}')
# print('=' * 20)

cand_sparsity = list()
cand_bits = list()
for arch in candidates:
    complexity = get_net_info(arch, config)
    cand_sparsity.append(float(complexity['sparsity']))
    cand_bits.append(complexity['bits'])

# for b, s in zip(cand_bits, cand_sparsity):
#     print(f'b : {b:.2f}, s : {s:.2f}')

# print(f'cand_sparsity : {cand_sparsity}')
# print(f'cand_bits : {cand_bits}')
plt.scatter(cand_bits, cand_sparsity, s=5)
plt.xlabel('bits')
plt.ylabel('layer sparsity')
plt.title(f'iter {iter}')

plt.show()
plt.savefig(fig_path, dpi=300)