import json
import numpy as np

N_BLK=32
N_LINEAR=7

dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_1000.json'

linear_numel = [16777216, 16777216, 16777216, 16777216, 45088768, 45088768, 45088768]
model_numel = sum(linear_numel) * N_BLK

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

x_set = [np.fromstring(k, sep=' ', dtype=int).reshape(N_BLK, N_LINEAR) for k in dataset.keys()]
y_set = list(dataset.values())

# print(f'x : {x}')
# print(f'y : {y}')

linears = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]

output = list()

for x, y in zip(x_set, y_set):
    arch = dict()
    memory_usage = 0
    for i, linear in enumerate(linears):
        arch[linear] = np.where(x[:, i] == 0, 2, 4).tolist()
        memory_usage += sum(arch[linear]) * linear_numel[i]
    bits = memory_usage / model_numel
    output.append([arch, y, bits])


new_dataset_path = '/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_ppl_1000.json'
with open(new_dataset_path, 'w') as f:
    json.dump({'archive': output}, f, ensure_ascii=False, indent=4)

    
# {archive = [ [ {'self_attn.q_proj': [2, 4, ....]}, 12.4, 3.3 ], [], ... ]}
# {archive = [ [ arch, metric, constraints ], ..., ]}