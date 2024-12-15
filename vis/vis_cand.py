import os
import json
from utils.func import get_net_info
import matplotlib.pyplot as plt
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# model_name = 'Llama-2-7b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411211754_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0'
# iter_list = list(range(1, 300))
# # iter_list = list(range(1, 300, 10))
# fig_path=f'fig/{model_name}_cand.png'

# model_name = 'Llama-2-13b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411211811_{model_name}_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# iter_list = list(range(1, 450))
# fig_path=f'fig/{model_name}_cand.png'

model_name = 'Llama-2-7b-hf'
arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270816_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# iter_list = list(range(1, 300))
iter_list = [299]
# fig_path=f'fig/{model_name}_owq_cand.png'
fig_path=f'fig/{model_name}_owq_bar.png'

# model_name = 'Llama-2-13b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270821_{model_name}_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# iter_list = list(range(1, 450))
# fig_path=f'fig/{model_name}_owq_cand.png'


# iter_list = [1, 50, 100, 150, 200, 250, 299]
# iter_list = list(range(1, 300))

config_path = 'config/llama.json'
with open(config_path, 'r') as f:
    config = json.load(f)[model_name]

# linear_idx_list = np.array(range(int(config['n_linear']) * int(config['n_block'])))

# avg_linear_bits_list = list()
# avg_linear_bits_idx_list = list()
# for iter in iter_list:

#     with open(os.path.join(arch_folder, f'iter_{iter}.stats'), 'r') as f:
#         arch = json.load(f)
#     # archive = [a[0] for a in arch['archive']]
#     # archive = [a[0] for a in arch['archive']] + [a[0] for a in arch['candidates']]
#     candidates = [a[0] for a in arch['candidates']]

#     cand_arch = list()
#     cand_avg_bits = list()
#     cand_arch_bits_idx = list()
#     cand_linear_bits = {l: [] for l in config['linear']}
#     cand_linear_bits_idx = {l: [] for l in config['linear']}
#     n_arch = len(candidates)

#     for arch in candidates:
#         arch_concat = np.concatenate(list(arch['linear'].values()))
#         arch_concat_bits_idx = np.array([0 if a == 2 else 1 if a > 2 and a < 3 else 2 if a == 3 else 3 if a > 3 and a < 4 else 4 for a in arch_concat])
#         bits = get_net_info(arch, config)['bits']
#         cand_arch.append(arch_concat)
#         cand_arch_bits_idx.append(arch_concat_bits_idx)
#         cand_avg_bits.append(bits)
#         # for linear in config['linear']:
#         #     cand_linear_bits[linear].append()

#     cand_arch = np.stack(cand_arch, axis=1)
#     cand_arch_bits_idx = np.stack(cand_arch_bits_idx, axis=1)

#     avg_linear_bits = cand_arch.mean(axis=1)
#     avg_linear_bits_idx = cand_arch_bits_idx.mean(axis=1)

#     avg_linear_bits_list.append(avg_linear_bits)
#     avg_linear_bits_idx_list.append(avg_linear_bits_idx)

#     print(f'iter : {iter}')
#     print(f'cand_arch : {cand_arch.shape}')
#     print(f'cand_arch_bits_idx : {cand_arch_bits_idx.shape}')
#     print(f'cand_avg_bits : {len(cand_avg_bits)}')
#     print(f'avg_linear_bits : {avg_linear_bits.shape}')

import pdb; pdb.set_trace()

subnets, metric, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
sort_idx = np.argsort(metric)
F = np.column_stack((metric, sec_obj))[sort_idx, :]
# front = NonDominatedSorting().do(F, only_non_dominated_front=True)
# pf = F[front, :]
# ps = np.array(subnets)[sort_idx][front]
range_idx = np.argwhere(np.logical_and(F[:, 1] > args.target_bits_range[0], F[:, 1] < args.target_bits_range[1])).flatten()
pf = F[range_idx, :]
ps = np.array(subnets)[sort_idx][range_idx]


avg_linear_bits_list = np.stack(avg_linear_bits_list, axis=0)
avg_linear_bits_idx_list = np.stack(avg_linear_bits_idx_list, axis=0)

print(f'avg_linear_bits_list : {avg_linear_bits_list.shape}')
print(f'avg_linear_bits_idx_list : {avg_linear_bits_idx_list.shape}')

plt_linear_idx_list = np.tile(linear_idx_list, [len(iter_list), 1])
plt_iter_list = np.tile(iter_list, [len(linear_idx_list), 1]).T

# fig = plt.figure(figsize=(6, 6))

# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), subplot_kw={"projection":"3d"})
fig, axs = plt.subplots(nrows=1, ncols=1)
im = axs.matshow(avg_linear_bits_list.T)
# im = axs.matshow(avg_linear_bits_idx_list.T)
axs.set_xlabel('Iteration')
axs.set_ylabel('Linear Index')
axs.set_title(model_name)
plt.colorbar(im, location='right', shrink=0.5)


plt.bar()
plt.bar()


# plt.savefig(fig_path, dpi=300)

