import matplotlib.pyplot as plt
import json
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import os

iter = 20
fig_path = f'fig/quant/prior_front_13b.png'

# no_prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2412111241_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2411211754_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0'

# no_prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2412111240_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
# prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2411211811_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'

no_prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2412111240_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2411211811_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'

# no_prior_7b_archive_list = []

def get_archive_list(path, iter):
    with open(os.path.join(path, f'iter_{iter}.stats'), 'r') as f:
        json_file = json.load(f)
        archive = json_file['archive'] + json_file['candidates']
    metric, sec_obj =  [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return np.array(archive)[sort_idx][front]

iter_1 = 1
no_prior_13b_archive_1 = get_archive_list(no_prior_13b_path, iter_1)
prior_13b_archive_1 = get_archive_list(prior_13b_path, iter_1)

iter_2 = 30
no_prior_13b_archive_2 = get_archive_list(no_prior_13b_path, iter_2)
prior_13b_archive_2 = get_archive_list(prior_13b_path, iter_2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0., wspace=0.15)

axes[0].scatter([a[2] for a in no_prior_13b_archive_1], [a[1] for a in no_prior_13b_archive_1], s=3, alpha=0.5, label='w/o prior')
axes[0].scatter([a[2] for a in prior_13b_archive_1], [a[1] for a in prior_13b_archive_1], s=3, alpha=0.5, label='w/ prior')
axes[0].set_xlabel('Bits')
axes[0].set_ylabel('Loss')
axes[0].set_yticks(np.array(range(0, 5)) / 5)
axes[0].set_ylim([None, 1])
axes[0].grid(c='0.8')
axes[0].set_title(f'Iteration {iter_1}')
axes[0].legend(loc="upper right")
axes[0].set_axisbelow(True)

axes[1].scatter([a[2] for a in no_prior_13b_archive_2], [a[1] for a in no_prior_13b_archive_2], s=3, alpha=0.5, label='w/o prior')
axes[1].scatter([a[2] for a in prior_13b_archive_2], [a[1] for a in prior_13b_archive_2], s=3, alpha=0.5, label='w/ prior')
axes[1].set_xlabel('Bits')
# axes[1].set_ylabel('Loss')
axes[1].set_yticks(np.array(range(0, 5)) / 5)
axes[1].set_ylim([None, 1])
axes[1].grid(c='0.8')
axes[1].set_title(f'Iteration {iter_2}')
axes[1].legend(loc="upper right")
axes[1].set_axisbelow(True)

plt.legend()
plt.show()

plt.show()
plt.savefig(fig_path, dpi=300)