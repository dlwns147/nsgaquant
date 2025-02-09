import numpy as np
import json
from predictor.factory import get_predictor
from search_space.llama import LlamaQuantSearchSpace
import matplotlib.pyplot as plt

model_name = 'Llama-2-7b-hf'
# trainset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_ppl_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
# testset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_ppl_250_2_4_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
# trainset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_loss_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
# testset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_loss_250_range_2_4_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json'
trainset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_hqq_loss_1000_2_4_bits_2_4.json'
testset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_hqq_loss_1000_2_4_bits_2_4_test.json'
config_path = '/NAS/SJ/nsgaquant/config/llama.json'

predictor_type = 'rbf'
# predictor_type = 'mlp'
# predictor_type = 'gp'
# predictor_type = 'carts'

fig_path = f'/NAS/SJ/nsgaquant/fig/test_predictor_{predictor_type}.png'

with open(config_path, 'r') as f:
    config = json.load(f)[model_name]

with open(trainset_path, 'r') as f:
    trainset = json.load(f)['archive']
    # trainset = [[{'linear': x[0]}, x[1], x[2]] for x in trainset]

with open(testset_path, 'r') as f:
    testset = json.load(f)['archive']
    # testset = [[{'linear': x[0]}, x[1], x[2]] for x in testset]

pass_linear_list = []
for linear, linear_bits in trainset[0][0]['linear'].items():
    for blk, bits in enumerate(linear_bits):
        if bits == 4:
            pass_linear_list.append(f'{blk}.{linear}')

outlier_bits = {l: [] for l in config['linear']}

search_space = LlamaQuantSearchSpace(
    n_block=config['n_block'],
    quant_model_bits=[2, 3, 4],
    sec_obj='bits',
    sec_obj_range=[2, 4],
    config=config,
    outlier_bits=outlier_bits,
    pass_linear_list=pass_linear_list
)

train_inputs = np.array([search_space.encode_predictor(x[0]) for x in trainset])
train_targets = np.array([x[1] for x in trainset])

test_inputs = np.array([search_space.encode_predictor(x[0]) for x in testset])
test_targets = np.array([x[1] for x in testset])

print(f'loaded data')

n_block = config['n_block']
n_linear = config['n_linear']
lb = np.zeros((n_block, n_linear))
ub = np.ones((n_block, n_linear))

for linear_idx, linear in enumerate(config['linear']):
    ub[:, linear_idx] = len(getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1

lb, ub = lb.flatten(), ub.flatten()

lb = np.delete(lb, search_space.pass_linear_idx_list, axis=-1)
ub = np.delete(ub, search_space.pass_linear_idx_list, axis=-1)

kwargs = {'lb': lb, 'ub': ub}
# print(f'lb : {lb.shape}, ub : {ub.shape}')

import pdb; pdb.set_trace()
metric_predictor = get_predictor(predictor_type, train_inputs, train_targets, device='cpu', **kwargs)
train_output = metric_predictor.predict(train_inputs)
test_output = metric_predictor.predict(test_inputs)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

axes[0].scatter(train_targets, train_output, s=5, label='trainset', c='green')
axes[0].set_title('Train data')
axes[0].set_xlabel('targets')
axes[0].set_ylabel('outputs')
axes[0].grid(c='0.8')
axes[0].axline((0, 0), slope=1, c='gray')
# axes[0].set_xlim([None, 20])
# axes[0].set_ylim([None, 20])
# axes[0].legend(loc='upper right')

axes[1].scatter(test_targets, test_output, s=5, label='testset', c='green')
axes[1].set_title('Test data')
axes[1].set_xlabel('targets')
axes[1].set_ylabel('outputs')
axes[1].grid(c='0.8')
axes[1].axline((0, 0), slope=1, c='gray')
# axes[1].set_xlim([None, 20])
# axes[1].set_ylim([None, 20])
# axes[1].legend(loc='upper right')

fig.tight_layout() 
plt.savefig(fig_path, dpi=300)
