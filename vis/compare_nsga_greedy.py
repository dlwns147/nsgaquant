import csv
import matplotlib.pyplot as plt
import json


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


plt.matshow()