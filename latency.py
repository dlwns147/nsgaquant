import json
import torch
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import gc

from evaluator import LlamaEvaluator
from hqq.utils.patching import prepare_for_inference

from utils.eval_utils import measure_latency, eval_metric
from utils.func_utils import get_net_info
from utils.data_utils import get_loader
from model.skip_llama import block_replace

import csv
import math

def main(args):
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    evaluator = LlamaEvaluator(
        config=config,
        model_name=args.model_name,
        method=args.method,
    )

    with open(args.data, 'r') as f:
        archive = json.load(f)['archive']
        archs = [x[0] for x in archive]

    # archive = list()
    # for arch in archs:
    #     model = evaluator.sample(arch)
    #     latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=evaluator.device, iteration=args.iteration)
    #     sparsity = get_net_info(arch, config, args.method)['sparsity']
    #     archive.append([arch, latency, sparsity])
    #     print(f'arch : {arch}, lat : {latency:.2f}, sparsity : {sparsity}')
    #     if args.latency_file:
    #         with open(args.latency_file, 'w') as f:
    #             json.dump({'archive': archive}, f, ensure_ascii=False, indent=4)

    # exit()

    measure_latency(model, generation=True, batch_size=args.batch_size, device=evaluator.device, iteration=args.iteration, max_time=args.max_time)

    n_blk = 32
    arch = {'self_attn': [1] * n_blk, 'mlp': [1] * n_blk}
    model = evaluator.sample(arch)
    full_lat = measure_latency(model, generation=True, batch_size=args.batch_size, device=evaluator.device, iteration=args.iteration, max_time=args.max_time)
    archive = []
    for i in range(1, n_blk + 1):
        while True:
            temp_arch = np.ones((n_blk))
            temp_arch[-i:] = 0
            np.random.shuffle(temp_arch)
            arch = {'self_attn': temp_arch.tolist(), 'mlp': [1] * n_blk}

            model = evaluator.sample(arch)
            lat = measure_latency(model, generation=True, batch_size=args.batch_size, device=evaluator.device, iteration=args.iteration, max_time=args.max_time)
            if math.isclose(lat, args.max_time):
                continue

            lat = full_lat - lat
            archive.append([{'self_attn': arch['self_attn']}, lat])
            print(f"{arch['self_attn']}, {lat:.2f}")
            if args.latency_file:
                with open(args.latency_file, 'w') as f:
                    json.dump({'archive': archive}, f, ensure_ascii=False, indent=4)
            break

    arch = {'self_attn': [1] * n_blk, 'mlp': [1] * n_blk}
    for i in range(1, n_blk + 1):
        while True:
            temp_arch = np.ones((n_blk))
            temp_arch[-i:] = 0
            np.random.shuffle(temp_arch)
            arch = {'self_attn': [1] * n_blk, 'mlp': temp_arch.tolist()}

            model = evaluator.sample(arch)
            lat = measure_latency(model, generation=True, batch_size=args.batch_size, device=evaluator.device, iteration=args.iteration, max_time=args.max_time)
            if math.isclose(lat, args.max_time):
                continue

            lat = full_lat - lat
            archive.append([{'mlp': arch['mlp']}, lat])
            print(f"{arch['mlp']}, {lat:.2f}")
            if args.latency_file:
                with open(args.latency_file, 'w') as f:
                    json.dump({'archive': archive}, f, ensure_ascii=False, indent=4)
            break


    exit()

    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', low_cpu_mem_usage=True, torch_dtype='auto')


    # for i in range(len(model.model.layers) - 1, 0, -1):
    #     del model.model.layers[i]
    # block_replace(model)
    # print(model)
    # device = next(model.parameters()).device
    # one_block_model_latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=device, iteration=args.iteration)
    # # ppl = eval_metric(model, metric='ppl', loader=loader, device=device, seqlen=2048)
    # # del loader
    # gc.collect()
    # torch.cuda.empty_cache()
    # memory_usage = torch.cuda.memory_allocated() / 1024. / 1024. / 1024. 
    # # complexity = get_net_info(arch, config)

    # print(f'{arch} / {args.sec_obj}: {complexity[args.sec_obj]:.3f} / Latency: {latency} / memory_usage: {memory_usage} GB')
    # print(f'Latency: {latency} / ppl : {ppl:.2f} / memory_usage: {memory_usage} GB')
    # print(f'One Block Model Latency: {one_block_model_latency} / memory_usage: {memory_usage} GB')

    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', low_cpu_mem_usage=True, torch_dtype='auto')
    device = next(model.parameters()).device
    # for i in range(len(model.model.layers) - 1, 0, -1):
    #     del model.model.layers[i]
    block_replace(model)
    model.model.layers[0].skip_mlp(reuse=False)
    print(model)
    one_attn_model_latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=device, iteration=args.iteration)
    gc.collect()
    torch.cuda.empty_cache()
    memory_usage = torch.cuda.memory_allocated() / 1024. / 1024. / 1024. 
    print(f'One Attn Layer Model Latency: {one_attn_model_latency} / memory_usage: {memory_usage} GB')

    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', low_cpu_mem_usage=True, torch_dtype='auto')
    device = next(model.parameters()).device
    # for i in range(len(model.model.layers) - 1, 0, -1):
    #     del model.model.layers[i]
    block_replace(model)
    model.model.layers[0].skip_attn(reuse=False)
    print(model)
    one_mlp_model_latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=device, iteration=args.iteration)
    gc.collect()
    torch.cuda.empty_cache()
    memory_usage = torch.cuda.memory_allocated() / 1024. / 1024. / 1024. 
    print(f'One Mlp Layer Model Latency: {one_mlp_model_latency} / memory_usage: {memory_usage} GB')

    
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', low_cpu_mem_usage=True, torch_dtype='auto')
    device = next(model.parameters()).device
    full_model_latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=device, iteration=10)
    gc.collect()
    torch.cuda.empty_cache()
    memory_usage = torch.cuda.memory_allocated() / 1024. / 1024. / 1024. 
    print(f'Full Model Latency: {full_model_latency} / memory_usage: {memory_usage} GB')

    # attn_latency = one_block_model_latency - one_mlp_model_latency
    # mlp_latency = one_block_model_latency - one_attn_model_latency
    # etc_latency = one_block_model_latency - (attn_latency + mlp_latency)
    attn_latency = full_model_latency - one_mlp_model_latency
    mlp_latency = full_model_latency - one_attn_model_latency
    etc_latency = full_model_latency - len(model.model.layers) * (attn_latency + mlp_latency)

    appr_full_model_latency = etc_latency + len(model.model.layers) * (attn_latency + mlp_latency)
    print(f'attn_latency : {attn_latency}, mlp_latency : {mlp_latency}, etc_latency : {etc_latency}')
    print(f'appr_full_model_latency : {appr_full_model_latency}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--arch_file', type=str, default='',
                        help='file path to measure latency')
    parser.add_argument('--method', type=str, default='',
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    # parser.add_argument('--n_data', type=int, default=1000,
    #                     help='test batch size for inference')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--iteration', type=int, default=10,
                        help='')
    parser.add_argument('--nan_value', type=float, default=50,
                        help='')
    parser.add_argument('--metric', type=str, nargs='+', default=[], 
                        help="'latency', 'ppl'")
    parser.add_argument('--latency_file', type=str, default='',
                        help='')
    parser.add_argument('--data', type=str, default='',
                        help='')
    parser.add_argument('--max_time', type=int, default=1e9,
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

