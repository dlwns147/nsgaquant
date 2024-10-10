from acc_predictor.factory import get_acc_predictor
import json
import numpy as np
import torch

from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from copy import deepcopy
from eval_utils import eval_ppl, get_tokenizer, get_loaders
import gc

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


# x = np.array(list(train_dataset.keys()))
train_x = np.array([np.fromstring(k, sep=' ', dtype=int) for k in train_dataset.keys()])
train_y = np.array(list(train_dataset.values()))

test_x = np.array([np.fromstring(k, sep=' ', dtype=int) for k in test_dataset.keys()])
test_y = np.array(list(test_dataset.values()))
acc_predictor = get_acc_predictor(predictor_type, train_x, train_y)

dataset_3bit_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_arch_2.99_3.01_bits_10000.json'
with open(dataset_3bit_path, 'r') as f:
    dataset_3bit = json.load(f)

test_3bit_x = np.array([np.fromstring(k, sep=' ', dtype=int) for k in dataset_3bit.keys()])
test_3bit_y = np.array(list(dataset_3bit.values()))

out_list = []
for data, label in zip(test_3bit_x, test_3bit_y):
    out = acc_predictor.predict(data).item()
    print(f'label : {label:.3f}, out : {out:.3f}')
    out_list.append(out)

top_k = 10
topk_idx = np.argsort(out_list)[:top_k]
print(f'topk_idx : {topk_idx}')
# for idx in topk_idx:
print([[test_3bit_x[idx], test_3bit_y[idx], out_list[idx]] for idx in topk_idx])

model_path = "meta-llama"
model_name = "Llama-2-7b-hf"
# model_name = "Llama-2-13b-hf"
# model_name = "Llama-2-70b-hf"
model_id  = f'{model_path}/{model_name}'

dataset = 'wikitext2'
device = torch.device('cuda:0')

bits_list = [test_3bit_y[idx] for idx in topk_idx]
topk_out_list = [out_list[idx] for idx in topk_idx]

num_samples = 1000
n_blocks = 32
seqlen = 2048
num_layers = 2 * n_blocks

model_2bit_path = '/SSD/hqq/Llama-2-7b-hf_2bit_64gs_1axis'
model_4bit_path = '/SSD/hqq/Llama-2-7b-hf_4bit_128gs_1axis'

llama_linears = {'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'mlp': ['up_proj', 'gate_proj', 'down_proj']}
llama_linears = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']

_, testloader = get_loaders(dataset, seed=0, seqlen=seqlen, tokenizer=get_tokenizer(model_id))

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model.to(device)

quant_model_bit2 = AutoHQQHFModel.from_quantized(model_2bit_path)
quant_model_bit2.to(device)

quant_model_bit4 = AutoHQQHFModel.from_quantized(model_4bit_path)
quant_model_bit4.to(device)

ppl_list = list()
for idx in topk_idx:
    arch = test_3bit_x[idx].reshape(n_blocks, 7)
    for blk_idx, blk_arch in enumerate(arch):
        for linear_idx, linear_arch in enumerate(blk_arch):
            module, linear = llama_linears[linear_idx].split('.')
            if linear_arch == 0:
                setattr(getattr(model.model.layers[blk_idx], module), linear, deepcopy(getattr(getattr(quant_model_bit2.model.layers[blk_idx], module), linear)))
            elif linear_arch == 1:
                setattr(getattr(model.model.layers[blk_idx], module), linear, deepcopy(getattr(getattr(quant_model_bit4.model.layers[blk_idx], module), linear)))
    gc.collect()

    with torch.no_grad():
        ppl = eval_ppl(model, testloader, device=device)
    torch.cuda.empty_cache()
    ppl_list.append(ppl)
    # ppl_list.append(idx)

# import csv
# csv.save()
print(f'bits_list : {bits_list}')
print(f'topk_out_list : {topk_out_list}')
print(f'ppl_list : {ppl_list}')
plt.scatter(topk_out_list, ppl_list, s=5)
plt.title('3-bit Model Top-k PPL')
plt.xlabel('Predicted PPL')
plt.ylabel('Real PPL')
plt.show()
plt.savefig('test.png', dpi=300)
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

