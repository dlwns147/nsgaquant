from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import numpy as np
import matplotlib.pyplot as plt

from utils.data import get_loader
from model.skip_llama import block_replace, skip_mlp, use_mlp
import numpy.ma as ma

model_id = '/SSD/huggingface/meta-llama/Llama-2-13b-hf'
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True, output_hidden_states=True)
model = block_replace(model)
loader = get_loader('wikitext2', model=model_id, n_sample=128, train=True, seed=0, seqlen=2048)

random_input = torch.randint(0, 31999, (1, 2048), dtype=torch.long).to(model.device)
# fig_path = 'fig/layer/cos_sim_block.png'
fig_path = 'fig/layer/cos_sim_layer.png'
n_block = 40
n_layer = n_block * 2

func = nn.CosineSimilarity(dim=0, eps=1e-08)
n_batch = len(loader)

# cos_sim = np.zeros((n_block, n_block))
# mask = np.empty((n_block, n_block))
# hidden_states = []
# n_batch = len(loader)
# for inputs in tqdm(loader):
#     with torch.no_grad():
#         outputs = model(inputs.to(model.device))
#     for i in range(n_block):
#         for j in range(n_block):
#             if i <= j:
#                 sim = func(hidden_states[i], hidden_states[j])
#                 cos_sim[i, j] += sim
#                 mask[i, j] = False
#             else:
#                 mask[i, j] = True

#     gc.collect()
#     torch.cuda.empty_cache()
#     break
# del model
# gc.collect()
# torch.cuda.empty_cache()
# fig, axs = plt.subplots(nrows=1, ncols=1)
# im = axs.matshow(maskded_cos_sim, vmin=0., vmax=1., cmap="viridis_r")
# axs.set_xlabel('Block i')
# axs.set_ylabel('Block j')
# axs.set_title("Llama 2 13B")
# plt.colorbar(im, location='right', shrink=1.0)

cos_sim = np.zeros((n_layer, n_layer))
mask = np.empty((n_layer, n_layer))
attn_hidden_states = []
# for inputs in tqdm(loader):
#     with torch.no_grad():
#         outputs = model(inputs.to(model.device))
#     mlp_hidden_states = [o.flatten() for o in outputs.hidden_states[1:]]
#     break

with torch.no_grad():
    outputs = model(random_input)
mlp_hidden_states = [o.flatten() for o in outputs.hidden_states[1:]]

mlp_hidden_states = torch.stack(mlp_hidden_states)

for i in range(n_block):
    if i > 0:
        use_mlp(model, i-1)
    skip_mlp(model, i)
    
    # for inputs in tqdm(loader):
    #     with torch.no_grad():
    #         outputs = model(inputs.to(model.device))
    #     attn_hidden_states.append(outputs.hidden_states[1+i].flatten())
    #     break

    with torch.no_grad():
        outputs = model(random_input)
    attn_hidden_states.append(outputs.hidden_states[1+i].flatten())

attn_hidden_states = torch.stack(attn_hidden_states)
hidden_states = torch.stack([attn_hidden_states, mlp_hidden_states]).transpose(0, 1).reshape(n_layer, -1)
for i in range(n_layer):
    for j in range(n_layer):
        if i <= j:
            sim = func(hidden_states[i], hidden_states[j])
            cos_sim[i, j] += sim
            mask[i, j] = False
        else:
            mask[i, j] = True
# cos_sim /= n_batch

maskded_cos_sim = ma.masked_array(cos_sim, mask=mask)

fig, axs = plt.subplots(nrows=1, ncols=1)
im = axs.matshow(maskded_cos_sim, vmin=0., vmax=1., cmap="viridis_r")
axs.set_xlabel('Layer i')
axs.set_ylabel('Layer j')
axs.set_title("Llama 2 13B")
plt.colorbar(im, location='right', shrink=1.0)

# plt.matshow(maskded_cos_sim, vmin=0., vmax=1.)
# plt.colorbar()
# plt.xlabel('Block')
# plt.ylabel('Block')
# plt.title('Llama 2 13B')
plt.savefig(fig_path, dpi=300)

