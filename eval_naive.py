import argparse
import time
from tqdm import tqdm
import csv
import os

import numpy as np
import torch
import gc
import json
import csv
import time
from accelerate import Accelerator

from evaluator import LlamaEvaluator
from utils.func import init_accelerator, clean_up, get_net_info
from utils.eval import load_and_eval_ppl, eval_zeroshot


def eval(args):

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=f'{args.model_path}/{args.model_name}',
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        group_size=args.group_size,
        datasets=args.datasets
    )

    n_block = config['n_block']

    ppl = 0
    # arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}, 'layer': {l: [1]* n_block for l in config['layer']}}
    arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}}
    
    # accelerator.print(f'arch : {arch}')


    with open(args.linear_sensitivity, 'r') as f:
        data = list(csv.reader(f))
        linear_list = data[0]
        linear_sensitivity = data[1]
        idx_list = np.argsort(linear_sensitivity)
    linear_list = [linear_list[i] for i in idx_list]

    arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}}

    bits_list = []
    for linear in linear_list:
        blk_idx, layer, linear = linear.split('.')
        blk_idx = int(blk_idx)

        arch['linear'][f'{layer}.{linear}'][blk_idx] = min(args.quant_model_bits)
        bits_list.append(abs(get_net_info(arch, config, group_size=args.group_size)['bits'] - args.target_bit))
    last_layer_idx = bits_list.index(min(bits_list))
    last_layer = linear_list[last_layer_idx]

    arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}}
    for linear in linear_list[:last_layer_idx + 1]:
        blk_idx, layer, linear = linear.split('.')
        blk_idx = int(blk_idx)

        arch['linear'][f'{layer}.{linear}'][blk_idx] = min(args.quant_model_bits)

    print(f'target_bit : {args.target_bit}, last_layer : {last_layer}, linear_sensitivity : {args.linear_sensitivity}')
    print(f'arch: {arch}')
    print(f'complexity: {get_net_info(arch, config, group_size=args.group_size)}')

    ppl, complexity = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl')
    # complexity = get_net_info(arch, config, group_size=args.group_size)
    print(f'bits : {complexity["bits"]}, ppl :{list(ppl.values())}')

    del evaluator
    clean_up()
    print(f'memory : {torch.cuda.memory_allocated()}')
    if args.zeroshot:
        from transformers import AutoTokenizer
        model_id = f'{args.model_path}/{args.model_name}'
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        results = eval_zeroshot(model, tokenizer=tokenizer, batch_size=args.zeroshot_batch_size, task_list=args.tasks)
        acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
        acc = [task_result['acc,none'] for task_result in results.values()]
        
        task = list(results.keys())
        avg_acc_norm = np.mean(acc_norm)
        avg_acc = np.mean(acc)
        print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        print(f'task : {task}')
        print(f'acc_norm : {acc_norm}')
        print(f'acc : {acc}')
    return 

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--group_size', type=int, default=-1,
                        help='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['wikitext2', 'c4'], 
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--outlier_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['piqa','winogrande','hellaswag','arc_challenge','arc_easy', 'lambada_openai', 'boolq'])
    parser.add_argument('--zeroshot_batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--linear_sensitivity', type=str, default='',
                        help='')
    parser.add_argument('--target_bit', type=float, default=3,
                        help='')

    cfgs = parser.parse_args()
    eval(cfgs)

