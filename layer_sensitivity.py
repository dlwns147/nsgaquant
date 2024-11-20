import argparse
import time
from tqdm import tqdm
import csv

import numpy as np
import torch
import gc
import json
import csv
import time
from accelerate import Accelerator

from evaluator import LlamaEvaluator


def layer_sensitivity(args):
    accelerator = Accelerator()
    accelerator.print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        model_name=args.model_name,
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        loss_func=args.loss_func,
        datasets=[args.dataset]
    )
    
    n_block = config['n_block']

    ppl = 0
    loss_list = dict()
    ppl_list = dict()
    arch = {'layer': {l: [1] * n_block for l in config['layer']}}
    
    for layer in config['layer']:
        for block_idx in range(n_block):
            ppl_list[layer] = 0
    accelerator.print(f'arch : {arch}')

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")

    # for linear in config['linear']:
    #     for block_idx in range(n_block):
    #         arch['linear'][linear][block_idx] = min(args.quant_model_bits)

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")

    # exit()
    start_point = time.time()
    
    for block_idx in range(n_block):
        for layer in config['layer']:
            iter_start = time.time()
            
            arch['layer'][layer][block_idx] = 0

            key = f'{block_idx}.{layer}'
            loss, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='loss', loss_func=args.loss_func)
            loss_list[key] = loss[args.dataset]
            if args.eval_ppl:
                ppl, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl', loss_func=args.loss_func)
                ppl_list[key] = ppl[args.dataset]
            iter_time = time.time() - iter_start
            accelerator.print(f"[{key} replaced] Loss={loss_list[key]:.4f}, PPL: {ppl_list[key]:.2f}, time: {iter_time:.2f}")
            
            arch['layer'][layer][block_idx] = 1

    with accelerator.main_process_first():
        if args.loss_csv_file:
            with open(args.loss_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(loss_list.keys()))
                write.writerow(list(loss_list.values()))
        if args.eval_ppl and args.ppl_csv_file:
            with open(args.ppl_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(ppl_list.keys()))
                write.writerow(list(ppl_list.values()))

    finish_point = time.time()
    time_elapsed = finish_point - start_point

    accelerator.print(f"Time_Elapsed: {time_elapsed}")
    accelerator.print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--loss_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--loss_func', type=str, default='cross_entropy', help='')

    cfgs = parser.parse_args()
    layer_sensitivity(cfgs)

