
from transformers import AutoModelForCausalLM, AutoConfig, HqqConfig
from utils.data import get_loader
from utils.eval import eval_metric, get_logits
from utils.func import hassubattr
from hqq.models.hf.base import AutoHQQHFModel
import json

from utils.data import get_loader
from utils.eval import eval_metric
from utils.func import getsubattr
from utils.dispatch import simple_dispatch_model

from accelerate import (
    Accelerator,
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
    load_checkpoint_and_dispatch
)

accelerator = Accelerator()

model_path = '/SSD/huggingface/meta-llama'
# model_name = 'Llama-2-7b-hf'
model_name = 'Llama-2-13b-hf'
model_id = f'{model_path}/{model_name}'

# fp16_model_path = '/SSD/huggingface/meta-llama/Llama-2-7b-hf'
fp16_model_path = f'{model_path}/{model_name}'
# fp16_model_path = '/SSD/huggingface/meta-llama/Meta-Llama-3-8B'

awq_4bit_mode_path = f'/SSD/awq/{model_name}_w4_g128_fake_3bit_awq.pt'
hqq_4bit_model_path = f'/SSD/hqq/{model_name}_4bit_128gs_1axis_qscale_false_qzero_false'

config_path = 'config/llama.json'
with open(config_path, 'r') as f:
    config = json.load(f)[model_name]

# config = AutoConfig.from_pretrained(fp16_model_path)

# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)

# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(fp16_model_path, torch_dtype='auto', device_map='cpu')
# model = AutoModelForCausalLM.from_pretrained(fp16_model_path, torch_dtype='auto', device_map='cpu')
# model = AutoModelForCausalLM.from_pretrained(awq_4bit_mode_path, torch_dtype='auto', device_map='cpu')
# model = AutoModelForCausalLM.from_pretrained(awq_4bit_mode_path, torch_dtype='auto', device_map='cpu')
# print(f'model.hf_device_map : {model.hf_device_map}')

model = AutoHQQHFModel.from_quantized(hqq_4bit_model_path, device_map='cpu')
# model = AutoHQQHFModel.from_quantized(hqq_4bit_model_path, device_map='auto')

bits=4
group_size=128
axis=1
hqq_config = HqqConfig(nbits=bits, group_size=group_size, axis=axis, initialize=True)
cache_dir = f'/SSD/hqq/cache/{model_id}_{bits}bit_{group_size}_{axis}axis'

# gpu_ids=['0', '1']
gpu_ids=['0']
gpu_start_idx = 0
n_proc = 1
gpu_per_proc = len(gpu_ids) // n_proc
n_block = int(config['n_block'])
assert n_block % gpu_per_proc == 0

blk_per_gpu = n_block // gpu_per_proc
cur_gpu_ids = list(range(gpu_start_idx, len(gpu_ids), n_proc))
print(f'cur_gpu_ids : {cur_gpu_ids}, blk_per_gpu : {blk_per_gpu}')
print(f'device : {accelerator.device}')

device_map = dict()
for pre_layer in config['pre_layer']:
    if hassubattr(model, pre_layer):
        device_map[pre_layer] = cur_gpu_ids[0]

for layer_idx in range(n_block):
    device_map[f"{config['layers']}.{layer_idx}"] = cur_gpu_ids[layer_idx // blk_per_gpu]
        
for post_layer in config['post_layer']:
    if hassubattr(model, post_layer):
        device_map[post_layer] = cur_gpu_ids[-1]

# print(f'model.hf_device_map : {model.hf_device_map}')
# device_map = model.hf_device_map
print(f'device_map : {device_map}')
# for module, device in device_map.items():
#     getsubattr(model, module).to(f'cuda:{device}')
# model.hf_device_map = hf_device_map
model = simple_dispatch_model(model, device_map=device_map)

# model = load_checkpoint_and_dispatch(model, checkpoint=cache_dir, device_map=device_map)


# model = AutoModelForCausalLM.from_pretrained(fp16_model_path, torch_dtype='auto', device_map=device_map, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(fp16_model_path, torch_dtype='auto', quantization_config=hqq_config, device_map=device_map, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(fp16_model_path, quantization_config=hqq_config, device_map='auto', cache_dir=cache_dir)
# quantization_config=hqq_config, 
# model = AutoHQQHFModel.from_quantized(hqq_4bit_model_path, device_map='auto')
# for n, p in model.named_parameters():
#     print(f'{n} : {p.device}')
print(f'model : {model.hf_device_map}')
# for n, m in model.named_modules():
#     if hasattr(m, 'meta'):
#         import pdb; pdb.set_trace()
model.eval()

loader = accelerator.prepare(get_loader('wikitext2', train=False, model=model_id))
ppl = eval_metric(model, accelerator=accelerator, metric='ppl', loader=loader, seqlen=2048)
print(f'ppl : {ppl}')
# exit()
