import os
import json
from utils.func import get_net_info
import matplotlib.pyplot as plt
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from time import time

# model_name = 'Llama-2-7b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411211754_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0'
# iter_list = list(range(1, 300))
# # iter_list = list(range(1, 300, 10))
# fig_path=f'fig/{model_name}_cand_3bits.png'

# model_name = 'Llama-2-13b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411211811_{model_name}_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# iter_list = list(range(1, 450))
# fig_path=f'fig/{model_name}_cand_3bits.png'

# model_name = 'Llama-2-7b-hf'
# arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270816_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# iter_list = list(range(1, 300))
# # iter_list = [299]
# fig_path=f'fig/{model_name}_owq_cand_3bits.png'

model_name = 'Llama-2-13b-hf'
arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270821_{model_name}_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
iter_list = list(range(1, 450))
# iter_list = [449]
fig_path=f'fig/{model_name}_owq_cand_3bits.png'


# iter_list = [1, 50, 100, 150, 200, 250, 299]
# iter_list = list(range(1, 300))

config_path = 'config/llama.json'
with open(config_path, 'r') as f:
    config = json.load(f)[model_name]

arch_list = list()
for iter in iter_list:
    start = time()
    
    with open(os.path.join(arch_folder, f'iter_{iter}.stats'), 'r') as f:
        arch = json.load(f)
    archive = arch['archive'] + arch['candidates']
    # candidates = [a[0] for a in arch['candidates']]

    subnets, metric, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    threshold = 0.005

    range_idx = np.argwhere(np.logical_and(F[:, 1] > (3 - threshold), F[:, 1] < (3 + threshold))).flatten()
    arch_list.append(np.array(subnets)[sort_idx][range_idx][0])
    iter_time = time() - start
    print(f'iter : {iter}, iter_time : {iter_time:.3f}')

arch_list = np.stack([np.concatenate(list(a['linear'].values())) for a in arch_list], axis=0)

arch_idx_list = np.zeros_like(arch_list)
arch_idx_list[np.where(arch_list == 2)] = 0
arch_idx_list[np.where(np.logical_and(arch_list > 2, arch_list < 3))] = 1
arch_idx_list[np.where(arch_list == 3)] = 2
arch_idx_list[np.where(np.logical_and(arch_list > 3, arch_list < 4))] = 3
arch_idx_list[np.where(arch_list == 4)] = 4

# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), subplot_kw={"projection":"3d"})
fig, axs = plt.subplots(nrows=1, ncols=1)
# im = axs.matshow(arch_list)
im = axs.matshow(arch_idx_list)

axs.set_xlabel('Linear Index')
axs.set_ylabel('Iteration')

axs.set_title(model_name)
plt.colorbar(im, location='right', shrink=0.5)

plt.savefig(fig_path, dpi=300)
