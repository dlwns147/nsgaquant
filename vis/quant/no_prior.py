import matplotlib.pyplot as plt
import json
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

fig_path = 'fig/quant/prior.png'

no_prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2412111241_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
no_prior_7b_iter_1_path = f'{no_prior_7b_path}/iter_1.stats'
no_prior_7b_iter_299_path = f'{no_prior_7b_path}/iter_299.stats'

prior_7b_path = '/NAS/SJ/nsgaquant/save/search/2411211754_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0'
prior_7b_iter_1_path = f'{prior_7b_path}/iter_1.stats'
prior_7b_iter_299_path = f'{prior_7b_path}/iter_299.stats'

no_prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2412111240_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
no_prior_13b_iter_1_path = f'{no_prior_13b_path}/iter_1.stats'
no_prior_13b_iter_449_path = f'{no_prior_13b_path}/iter_449.stats'

prior_13b_path = '/NAS/SJ/nsgaquant/save/search/2411211811_Llama-2-13b-hf_bits_loss_hqq_iter_450_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
prior_13b_iter_1_path = f'{prior_13b_path}/iter_1.stats'
prior_13b_iter_449_path = f'{prior_13b_path}/iter_449.stats'


with open(no_prior_7b_iter_1_path, 'r') as f:
    json_file = json.load(f)
    no_prior_7b_iter_1_archive = json_file['archive'] + json_file['candidates']

with open(no_prior_7b_iter_299_path, 'r') as f:
    json_file = json.load(f)
    no_prior_7b_iter_299_archive = json_file['archive'] + json_file['candidates']
    
    metric, sec_obj =  [v[1] for v in no_prior_7b_iter_299_archive], [v[2] for v in no_prior_7b_iter_299_archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    no_prior_7b_iter_299_archive = np.array(no_prior_7b_iter_299_archive)[sort_idx][front]

with open(prior_7b_iter_1_path, 'r') as f:
    json_file = json.load(f)
    prior_7b_iter_1_archive = json_file['archive'] + json_file['candidates']

with open(prior_7b_iter_299_path, 'r') as f:
    json_file = json.load(f)
    prior_7b_iter_299_archive = json_file['archive'] + json_file['candidates']

    metric, sec_obj =  [v[1] for v in prior_7b_iter_299_archive], [v[2] for v in prior_7b_iter_299_archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    prior_7b_iter_299_archive = np.array(prior_7b_iter_299_archive)[sort_idx][front]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0, 0].scatter([a[2] for a in no_prior_7b_iter_1_archive], [a[1] for a in no_prior_7b_iter_1_archive], s=3, alpha=0.5, label='w/o prior')
axes[0, 0].scatter([a[2] for a in prior_7b_iter_1_archive], [a[1] for a in prior_7b_iter_1_archive], s=3, alpha=0.5, label='w prior')
axes[0, 0].set_xlabel('Bits')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Llama-2-7b Iteration 1')
axes[0, 0].legend(loc="upper right")

axes[0, 1].scatter([a[2] for a in no_prior_7b_iter_299_archive], [a[1] for a in no_prior_7b_iter_299_archive], s=3, alpha=0.5, label='w/o prior')
axes[0, 1].scatter([a[2] for a in prior_7b_iter_299_archive], [a[1] for a in prior_7b_iter_299_archive], s=3, alpha=0.5, label='w prior')
axes[0, 1].set_xlabel('Bits')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_ylim([None, 1.5])
axes[0, 1].set_title('Llama-2-7b Iteration 300')
axes[0, 1].legend(loc="upper right")


with open(no_prior_13b_iter_1_path, 'r') as f:
    json_file = json.load(f)
    no_prior_13b_iter_1_archive = json_file['archive'] + json_file['candidates']

with open(no_prior_13b_iter_449_path, 'r') as f:
    json_file = json.load(f)
    no_prior_13b_iter_449_archive = json_file['archive'] + json_file['candidates']
    
with open(prior_13b_iter_1_path, 'r') as f:
    json_file = json.load(f)
    prior_13b_iter_1_archive = json_file['archive'] + json_file['candidates']

axes[1, 0].scatter([a[2] for a in no_prior_13b_iter_1_archive], [a[1] for a in no_prior_13b_iter_1_archive], s=3, alpha=0.5, label='w/o prior')
axes[1, 0].scatter([a[2] for a in prior_13b_iter_1_archive], [a[1] for a in prior_13b_iter_1_archive], s=3, alpha=0.5, label='w prior')
axes[1, 0].set_xlabel('Bits')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Llama-2-13b Iteration 1')
axes[1, 0].legend(loc="upper right")

# axes[1, 1].scatter([a[2] for a in no_prior_7b_iter_299_archive], [a[1] for a in no_prior_7b_iter_299_archive], s=3, alpha=0.5, label='w/o prior')
# axes[1, 1].scatter([a[2] for a in prior_7b_iter_299_archive], [a[1] for a in prior_7b_iter_299_archive], s=3, alpha=0.5, label='w prior')
# axes[1, 1].set_xlabel('Bits')
# axes[1, 1].set_ylabel('Loss')
# axes[1, 1].set_ylim([None, 1.5])
# axes[1, 1].set_title('Llama-2-7b Iteration 449')
# axes[1, 1].legend(loc="upper right")

plt.show()
plt.savefig(fig_path, dpi=300)