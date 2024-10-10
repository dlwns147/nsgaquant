import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_net_info
import os
from tqdm import tqdm 

ppl_arch_figure = '/NAS/SJ/nsgaquant/fig/predictor.png'

model_name='meta-llama/Llama-2-7b-hf'

results_folder = "save/search/Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_2410071303"
total_iter = 300
iteration_list = list(range(1, total_iter + 1))

hv_list = list()
rmse_list = list()
rho_list = list()
tau_list = list()

for iter in tqdm(iteration_list, desc='Reading result files'):
    with open(os.path.join(results_folder, f'iter_{iter}.stats')) as f:
        result = json.load(f)
        hv_list.append(result['hv'])
        rmse_list.append(result['surrogate']['rmse'])
        rho_list.append(result['surrogate']['rho'])
        tau_list.append(result['surrogate']['tau'])

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0].plot(iteration_list, hv_list)
axes[0].set_title('Hypervolume')
axes[0].set_xlabel('iteration')
axes[0].set_ylabel('hypervolume')
# axes[0].legend(loc='lower right')

axes[1].plot(iteration_list, rmse_list)
axes[1].set_title('RMSE')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('rmse')
# axes[1].legend(loc='upper right')

axes[2].plot(iteration_list, rho_list)
axes[2].set_title("Spearman Rho")
axes[2].set_xlabel('iteration')
axes[2].set_ylabel('rho')
# axes[2].legend(loc='lower right')

axes[3].plot(iteration_list, tau_list)
axes[3].set_title('Kendall Tau')
axes[3].set_xlabel('iteration')
axes[3].set_ylabel('tau')
# axes[3].legend(loc='lower right')

plt.show()
plt.savefig(ppl_arch_figure, dpi=300)