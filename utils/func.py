import numpy as np
from accelerate import Accelerator
from hqq.models.hf.base import AutoHQQHFModel
from .dispatch import simple_dispatch_model
import scipy.stats as stats
import json

def get_correlation(prediction, target):


    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau

def compute_latency(arch, config, latency_table):
    # with open(config['mpe_table_json'], 'r') as f:
    #     mpe_table = json.load(f)
    if latency_table is None:
        return 0
        
    latency = 0
    if 'fp16_model' in latency_table:
        latency = latency_table['fp16_model']
        for name in arch['linear']:
            for bit in arch['linear'][name]:
                latency += latency_table[str(int(bit))][name]
        return latency
    elif 'block' in latency_table:
        for blk_idx in range(int(config['n_block'])):
            if all([layer_arch[blk_idx] == 1 for layer_arch in arch['layer'].values()]):
                latency += latency_table['block']
            else:
                for layer, layer_arch in arch['layer'].items():
                    if layer_arch[blk_idx] == 1:
                        latency += latency_table[layer]
                        break

        latency += latency_table["etc"]
        return latency
        # return latency / latency_table["full"]

def compute_bits(arch, config):
    memory_usage = 0
    for linear_group, bits in arch['linear'].items():
        for blk, bit in enumerate(bits):
            for linear in linear_group.split(','):
                out_dim, in_dim = config['linear_shape'][linear]
                memory_usage += int(out_dim) * int(in_dim) * bit * arch['layer'][config['hierarchy'][linear]][blk]
    return memory_usage / config['model_numel']

def compute_sparsity(arch):
    return np.concatenate([v for v in arch['layer'].values()]).mean()

def compute_params(arch, config):
    params = 0
    total_params = 0
    for layer, layer_arch in arch['layer'].items():
        for layer_mask in layer_arch:
            total_params += config['layer_numel'][layer]
            params += config['layer_numel'][layer] * layer_mask
            
    return params / total_params

def get_net_info(arch, config, latency_table):
    net_info = {}
    net_info['bits'] = compute_bits(arch, config)
    net_info['sparsity'] = compute_sparsity(arch)
    net_info['params'] = compute_params(arch, config)
    net_info['latency'] = compute_latency(arch, config, latency_table) # ms
    
    return net_info

def getsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return getsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return getattr(obj, attr)
    
def setsubattr(obj, attr, value):
    attrs = attr.split('.')
    if len(attrs) > 1:
        setsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]), value)
    else :
        setattr(obj, attr, value)

def delsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return delsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return delattr(obj, attr)
    
def hassubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return hassubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return hasattr(obj, attr)

def getblock(model, config):
    return getsubattr(model, config['layers'])


def init_accelerator(gpu_id, config):
    gpu_id = gpu_id.split(',')
    accelerator = Accelerator()
    n_proc = accelerator.num_processes
    assert len(gpu_id) % n_proc == 0, 'Total number of gpus (args.gpu_id) should be divisible by num_processes'

    gpu_start_idx = accelerator.device.index if accelerator.device.index is not None else 0
    
    gpu_per_proc = len(gpu_id) // n_proc
    n_block = int(config['n_block'])
    assert n_block % gpu_per_proc == 0, f'n_block {n_block} is not divisible by {gpu_per_proc}'

    blk_per_gpu = n_block // gpu_per_proc
    cur_gpu_id = list(range(gpu_start_idx, len(gpu_id), n_proc))

    device_map = dict()
    for pre_layer in config['pre_layer']:
        device_map[pre_layer] = cur_gpu_id[0]

    for layer_idx in range(n_block):
        device_map[f"{config['layers']}.{layer_idx}"] = cur_gpu_id[layer_idx // blk_per_gpu]
            
    for post_layer in config['post_layer']:
        device_map[post_layer] = cur_gpu_id[-1]

    # print(f'cur_gpu_ids : {cur_gpu_id}, blk_per_gpu : {blk_per_gpu}, device : {accelerator.device}, device_map : {device_map}')
    # print(f'device_map : {device_map}')

    return accelerator, device_map

def load_hqq_model(model_id, device_map):
    model = AutoHQQHFModel.from_quantized(model_id, device_map='cpu')
    model = simple_dispatch_model(model, device_map)
    return model

def insert_fp16_channel_hqq(linear, outlier):
    # import pdb; pdb.set_trace()
    linear.meta['outlier'] = outlier

def remove_fp16_channel_hqq(linear):
    if 'outlier' in linear.meta:
        del linear.meta['outlier']

def get_fp16_channel(linear, idx):
    # print(f'linear.weight : {linear.weight.data.device}, idx : {idx}')
    return linear.weight.data[:, idx]

def get_outlier_bits(config):
    pass