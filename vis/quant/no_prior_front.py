import matplotlib.pyplot as plt
import json
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import os

iter = 20
fig_path = f'fig/quant/prior_front_iter{iter}.png'

no_prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2412111241_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2411211754_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0'

no_prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2412111240_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2411211811_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'

no_prior_7b_archive_list = []

def get_archive_list(path, iter):
    with open(os.path.join(path, f'iter_{iter}.stats'), 'r') as f:
        json_file = json.load(f)
        archive = json_file['archive'] + json_file['candidates']
    metric, sec_obj =  [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return np.array(archive)[sort_idx][front]

no_prior_7b_archive = get_archive_list(no_prior_7b_path, iter)
prior_7b_archive = get_archive_list(prior_7b_path, iter)

no_prior_13b_archive = get_archive_list(no_prior_13b_path, iter)
prior_13b_archive = get_archive_list(prior_13b_path, iter)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0].scatter([a[2] for a in no_prior_7b_archive], [a[1] for a in no_prior_7b_archive], s=3, alpha=0.5, label='w/o prior')
axes[0].scatter([a[2] for a in prior_7b_archive], [a[1] for a in prior_7b_archive], s=3, alpha=0.5, label='w prior')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('Loss')
axes[0].set_ylim([None, 1.5])
axes[0].set_title('Llama-2-7b')
axes[0].legend(loc="upper right")

axes[1].scatter([a[2] for a in no_prior_13b_archive], [a[1] for a in no_prior_13b_archive], s=3, alpha=0.5, label='w/o prior')
axes[1].scatter([a[2] for a in prior_13b_archive], [a[1] for a in prior_13b_archive], s=3, alpha=0.5, label='w prior')
axes[1].set_xlabel('Bits')
axes[1].set_ylabel('Loss')
axes[1].set_ylim([None, 1])
axes[1].set_title('Llama-2-13b')
axes[1].legend(loc="upper right")

plt.show()
plt.savefig(fig_path, dpi=300)