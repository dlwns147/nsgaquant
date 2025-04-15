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
from utils.func import init_accelerator, cleanup
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
        datasets=args.datasets
    )

    n_block = config['n_block']

    ppl = 0
    # arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}, 'layer': {l: [1]* n_block for l in config['layer']}}
    arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}}
    
    # accelerator.print(f'arch : {arch}')

    print(f'last_linear : {args.last_linear}, greedy_search_result : {args.greedy_search_result}')
    with open(args.greedy_search_result, 'r') as f:
        selected_linears = list(csv.reader(f))[0]
        selected_linears = selected_linears[:selected_linears.index(args.last_linear) + 1]
    for linear in selected_linears:
        blk_idx, layer, linear = linear.split('.')
        blk_idx = int(blk_idx)

        arch['linear'][f'{layer}.{linear}'][blk_idx] = min(args.quant_model_bits)


    ppl, complexity = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl')
    print(f'ppl :{ppl}, bits : {complexity["bits"]}')

    model = evaluator.sample(arch)
    del evaluator
    cleanup()
    print(f'memory : {torch.cuda.memory_allocated()}')
    
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
    parser.add_argument('--greedy_search_result', type=str, default='',
                        help='')
    parser.add_argument('--last_linear', type=str, default='',
                        help='')

    cfgs = parser.parse_args()
    eval(cfgs)

