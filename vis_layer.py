import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

sleb_128_7b_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_128_js.csv'
sleb_256_7b_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_256_js.csv'
sleb_512_7b_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_512_js.csv'

nsga_7b_path = '/NAS/SJ/nsgaquant/save/result/2411182131_layer_prune/results_arch.json' # save/search/2411182009_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_127.stats
# nsga_7b_path = '/NAS/SJ/nsgaquant/save/result/2411252027_layer_prune/results_arch.json' # save/search/2411182009_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_95.stats
# nsga_7b_path = '/NAS/SJ/nsgaquant/save/result/2411211555_layer_prune/results_arch.json' # save/search/2411182009_Llama-2-7b-hf_params_loss_layer_prune_iter_128_n_iter_32_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_63.stats

sleb_figure = 'fig/layer_prune_res.png'

with open(sleb_128_7b_path, 'r') as f:
    sleb_128_7b_result = list(csv.reader(f))
    sleb_128_7b_ppl = list(map(float, sleb_128_7b_result[1]))
    sleb_128_7b_param = list(map(float, sleb_128_7b_result[3]))
    sleb_128_7b_mask = list(map(float, sleb_128_7b_result[4]))

with open(sleb_256_7b_path, 'r') as f:
    sleb_256_7b_result = list(csv.reader(f))
    sleb_256_7b_ppl = list(map(float, sleb_256_7b_result[1]))
    sleb_256_7b_param = list(map(float, sleb_256_7b_result[3]))
    sleb_256_7b_mask = list(map(float, sleb_256_7b_result[4]))

with open(sleb_512_7b_path, 'r') as f:
    sleb_512_7b_result = list(csv.reader(f))
    sleb_512_7b_ppl = list(map(float, sleb_512_7b_result[1]))
    sleb_512_7b_param = list(map(float, sleb_512_7b_result[3]))
    sleb_512_7b_mask = list(map(float, sleb_512_7b_result[4]))


with open(nsga_7b_path, 'r') as f:
    nsga_archive = json.load(f)['archive']
    nsga_param = [a[1] for a in nsga_archive]
    nsga_ppl = [a[2]['wikitext2'] for a in nsga_archive]
    # sort_idx = np.argsort(nsga_ppl)
    F = np.column_stack((nsga_ppl, nsga_param))# [sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nsga_param = np.array(nsga_param)[front].tolist()
    nsga_ppl = np.array(nsga_ppl)[front].tolist()
    print(f'nsga_param : {nsga_param}')
    print(f'nsga_ppl : {nsga_ppl}')
    print(f'front : {front}')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0, 0].scatter(sleb_128_7b_param, sleb_128_7b_ppl, color='gray', s=3, label='sleb layer 128 (20m)')
axes[0, 0].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
axes[0, 0].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
axes[0, 0].scatter(nsga_param, nsga_ppl, color='blue', s=3, label='nsga (77m)')
axes[0, 0].set_title(f'Llama-2-7b')
axes[0, 0].set_xlabel('Params')
axes[0, 0].set_ylabel('PPL')
axes[0, 0].legend(loc="upper right")


axes[0, 1].scatter(sleb_128_7b_param, sleb_128_7b_ppl, color='gray', s=3, label='sleb layer 128 (20m)')
axes[0, 1].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
axes[0, 1].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
axes[0, 1].scatter(nsga_param, nsga_ppl, color='blue', s=3, label='nsga (77m)')
axes[0, 1].set_title(f'Llama-2-7b')
axes[0, 1].set_xlabel('Params')
axes[0, 1].set_ylabel('PPL')
axes[0, 1].set_xlim([0.8, 1.0])
axes[0, 1].set_ylim([5, 10])
axes[0, 1].legend(loc="upper right")

sleb_128_13b_path = '/NAS/SJ/sleb/csv/Llama-2-13b-hf_ppl_128_js.csv'
sleb_256_13b_path = '/NAS/SJ/sleb/csv/Llama-2-13b-hf_ppl_256_js.csv'
sleb_512_13b_path = '/NAS/SJ/sleb/csv/Llama-2-13b-hf_ppl_512_js.csv'

nsga_13b_path = '/NAS/SJ/nsgaquant/save/result/2411260852_layer_prune/results_arch.json' # save/search/2411251843_Llama-2-13b-hf_params_loss_layer_prune_iter_160_n_iter_40_nsga2_obj_0.5_1._jsd_mut_0.1_layer_prune_0.01_1.0/iter_159.stats

with open(sleb_128_13b_path, 'r') as f:
    sleb_128_13b_result = list(csv.reader(f))
    sleb_128_13b_ppl = list(map(float, sleb_128_13b_result[1]))
    sleb_128_13b_param = list(map(float, sleb_128_13b_result[3]))
    sleb_128_13b_mask = list(map(float, sleb_128_13b_result[4]))

with open(sleb_256_13b_path, 'r') as f:
    sleb_256_13b_result = list(csv.reader(f))
    sleb_256_13b_ppl = list(map(float, sleb_256_13b_result[1]))
    sleb_256_13b_param = list(map(float, sleb_256_13b_result[3]))
    sleb_256_13b_mask = list(map(float, sleb_256_13b_result[4]))

with open(sleb_512_13b_path, 'r') as f:
    sleb_512_13b_result = list(csv.reader(f))
    sleb_512_13b_ppl = list(map(float, sleb_512_13b_result[1]))
    sleb_512_13b_param = list(map(float, sleb_512_13b_result[3]))
    sleb_512_13b_mask = list(map(float, sleb_512_13b_result[4]))


with open(nsga_13b_path, 'r') as f:
    nsga_archive = json.load(f)['archive']
    nsga_param = [a[1] for a in nsga_archive]
    nsga_ppl = [a[2]['wikitext2'] for a in nsga_archive]
    # sort_idx = np.argsort(nsga_ppl)
    F = np.column_stack((nsga_ppl, nsga_param))# [sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nsga_param = np.array(nsga_param)[front].tolist()
    nsga_ppl = np.array(nsga_ppl)[front].tolist()
    print(f'nsga_param : {nsga_param}')
    print(f'nsga_ppl : {nsga_ppl}')
    print(f'front : {front}')

axes[1, 0].scatter(sleb_128_13b_param, sleb_128_13b_ppl, color='gray', s=3, label='sleb layer 128 (55m)')
axes[1, 0].scatter(sleb_256_13b_param, sleb_256_13b_ppl, color='red', s=3, label='sleb layer 256 (138m)')
axes[1, 0].scatter(sleb_512_13b_param, sleb_512_13b_ppl, color='purple', s=3, label='sleb layer 512 (246m)')
axes[1, 0].scatter(nsga_param, nsga_ppl, color='blue', s=3, label='nsga 128 (210m)')

axes[1, 0].set_title(f'Llama-2-13b')
axes[1, 0].set_xlabel('Params')
axes[1, 0].set_ylabel('PPL')
axes[1, 0].set_xlim([0.48, 1.02])
axes[1, 0].legend(loc="upper right")

axes[1, 1].scatter(sleb_128_13b_param, sleb_128_13b_ppl, color='gray', s=3, label='sleb layer 128 (55m)')
axes[1, 1].scatter(sleb_256_13b_param, sleb_256_13b_ppl, color='red', s=3, label='sleb layer 256 (138m)')
axes[1, 1].scatter(sleb_512_13b_param, sleb_512_13b_ppl, color='purple', s=3, label='sleb layer 512 (246m)')
axes[1, 1].scatter(nsga_param, nsga_ppl, color='blue', s=3, label='nsga 128 (210m)')
axes[1, 1].set_title(f'Llama-2-13b')
axes[1, 1].set_xlabel('Params')
axes[1, 1].set_ylabel('PPL')
axes[1, 1].set_xlim([0.8, 1.0])
axes[1, 1].set_ylim([4.5, 7])
axes[1, 1].legend(loc="upper right")


plt.show()
plt.savefig(sleb_figure, dpi=300)