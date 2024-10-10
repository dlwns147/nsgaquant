from acc_predictor.factory import get_acc_predictor
import json
import numpy as np
import torch

from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from copy import deepcopy
from eval_utils import eval_ppl, get_tokenizer, get_loaders

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# train_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_loss_uniform_1000.json'
# test_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_loss_uniform_test_1000.json'
train_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_1000.json'
test_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_test_1000.json'
predictor_type = 'mlp'

with open(train_dataset_path, 'r') as json_file:
    train_dataset = json.load(json_file)
with open(test_dataset_path, 'r') as json_file:
    test_dataset = json.load(json_file)

LLAMA_LINEARS_NUMEL = [4096 * 4096, 4096 * 4096, 4096 * 4096, 4096 * 4096, 4096 * 11008, 4096 * 11008, 11008 * 4096]
N_BLK = 32
N_LINEAR = 7
LLAMA_NUMEL = sum(LLAMA_LINEARS_NUMEL) * N_BLK

# x = np.array(list(train_dataset.keys()))
train_x = np.array([np.fromstring(k, sep=' ', dtype=int) for k in train_dataset.keys()])
train_y = np.array(list(train_dataset.values()))
train_bits = list()
train_x_reshape = train_x.reshape(-1, N_BLK, N_LINEAR)

for arch in train_x_reshape:
    memory_usage = 0
    for blk_arch in arch:
        for i, linear_arch in enumerate(arch):
            if linear_arch == 0:
                memory_usage += 2 * LLAMA_LINEARS_NUMEL[i]
            elif linear_arch == 1:
                memory_usage += 4 * LLAMA_LINEARS_NUMEL[i]

    cur_bits = memory_usage / LLAMA_NUMEL
    train_bits.append(cur_bits)


test_x = np.array([np.fromstring(k, sep=' ', dtype=int) for k in test_dataset.keys()])
test_y = np.array(list(test_dataset.values()))
test_bits = list()
test_x_reshape = test_x.reshape(-1, N_BLK, N_LINEAR)

for arch in test_x_reshape:
    memory_usage = 0
    for blk_arch in arch:
        for i, linear_arch in enumerate(arch):
            if linear_arch == 0:
                memory_usage += 2 * LLAMA_LINEARS_NUMEL[i]
            elif linear_arch == 1:
                memory_usage += 4 * LLAMA_LINEARS_NUMEL[i]

    cur_bits = memory_usage / LLAMA_NUMEL
    test_bits.append(cur_bits)

# acc_predictor = get_acc_predictor(predictor_type, train_x, train_y)

# print(f'===== train set ======')
# train_error_list = []
# train_out_list = []
# for data, label in zip(train_x, train_y):
#     out = acc_predictor.predict(data).item()
#     print(f'label : {label:.3f}, out : {out:.3f}')
#     error = abs(label - out)
#     train_out_list.append(out)
#     train_error_list.append(error)

# train_r2 = r2_score(train_y, train_out_list)

# test_error_list = []
# test_out_list = []

# print(f'===== test set ======')
# for data, label in zip(test_x, test_y):
#     out = acc_predictor.predict(data).item()
#     print(f'label : {label:.3f}, out : {out:.3f}')
#     error = abs(label - out)
#     test_out_list.append(out)
#     test_error_list.append(error)

# test_r2 = r2_score(test_y, test_out_list)
# print(f'train error : {sum(train_error_list):.3f}, mean : {np.mean(train_error_list):.3f}, std : {np.std(train_error_list):.3f}, r2 : {train_r2:.2f}')
# print(f'test error : {sum(test_error_list):.3f}, mean : {np.mean(test_error_list):.3f}, std : {np.std(test_error_list):.3f}, r2 : {test_r2:.2f}')


# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# ax[0].scatter(train_out_list, train_y, s=5)
# ax[0].set_title('PPL Train Dataset')
# ax[0].set_xlabel('Predicted PPL')
# ax[0].set_ylabel('Real PPL')
# # ax[0].text(0.5, 0.5, f'R^2: {train_r2}', fontsize=12)
# ax[0].set_xlim(left=5, right=12)
# ax[0].set_ylim(bottom=5, top=12)

# ax[1].scatter(test_out_list, test_y, s=5)
# ax[1].set_title('PPL Test Dataset')
# ax[1].set_xlabel('Predicted PPL')
# ax[1].set_ylabel('Real PPL')
# # ax[1].text(0.5, 0.5, f'R^2: {test_r2}', fontsize=12)
# ax[1].set_xlim(left=5, right=12)
# ax[1].set_ylim(bottom=5, top=12)

# ax[0].scatter(train_out_list, train_y, s=5)
# ax[0].set_title('Loss Train Dataset')
# ax[0].set_xlabel('Predicted Loss')
# ax[0].set_ylabel('Real Loss')
# # ax[0].text(0.5, 0.5, f'R^2: {train_r2}', fontsize=12)
# ax[0].set_xlim(left=10.2, right=14)
# ax[0].set_ylim(bottom=10.2, top=14)

# ax[1].scatter(test_out_list, test_y, s=5)
# ax[1].set_title('Loss Test Dataset')
# ax[1].set_xlabel('Predicted Loss')
# ax[1].set_ylabel('Real Loss')
# # ax[1].text(0.5, 0.5, f'R^2: {test_r2}', fontsize=12)
# ax[1].set_xlim(left=10.2, right=14)
# ax[1].set_ylim(bottom=10.2, top=14)

# plt.show()
# plt.savefig('test.png', dpi=300)

# print(f'train_error_list : {train_error_list}')
# print(f'test_error_list : {test_error_list}')
# print(f'train error : {sum(train_error_list)}, mean : {(sum(train_error_list) / len(train_error_list))}')
# print(f'test error : {sum(test_error_list)}, mean : {(sum(test_error_list) / len(test_error_list))}')

