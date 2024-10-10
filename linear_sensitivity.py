import argparse
import time
from tqdm import tqdm
import os
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from copy import deepcopy
import gc
import json
import csv
import time

# from search_space.llama import LlamaSearchSpace
from evaluator import LlamaEvaluator


def linear_sensitivity(args):
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device = torch.device("cuda:0")
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    evaluator = LlamaEvaluator(
        config=config,
        quant_method=args.quant_method,
        model_name=args.model_name,
        large_model_path=args.large_model_path,
        large_model_bits=args.large_model_bits,
        small_model_path=args.small_model_path,
        small_model_bits=args.small_model_bits,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset]
    )
    
    n_block = config['n_block']

    ppl = 0
    loss_list = dict()
    ppl_list = dict()
    arch = {l: [] for l in config['linear']}
    
    for linear in config['linear']:
        for block_idx in range(n_block):
            arch[linear].append(args.large_model_bits)
            ppl_list[f'{block_idx}.{linear}'] = 0

    start_point = time.time()

    for block_idx in range(n_block):
        for linear in config['linear']:
            iter_start = time.time()
            key = f'{block_idx}.{linear}'
            arch[linear][block_idx] = args.small_model_bits
            loss, _ = evaluator.eval(arch, metric='loss')
            loss_list[key] = loss[args.dataset]
            if args.eval_ppl:
                ppl, _ = evaluator.eval(arch, metric='ppl')
                ppl_list[key] = ppl[args.dataset]
            iter_time = time.time() - iter_start
            print(f"[{key} replaced] Loss={loss_list[key]:.4f}, PPL: {ppl_list[key]:.2f}, time: {iter_time:.2f}")
            
            arch[linear][block_idx] = args.large_model_bits
            gc.collect()
            torch.cuda.empty_cache()

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

    print(f"Time_Elapsed: {time_elapsed}")
    print(args)


def main(args):
    linear_sensitivity(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_bits', type=float, default=4,
                        help='test batch size for inference')
    parser.add_argument('--small_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--small_model_bits', type=float, default=2,
                        help='test batch size for inference')
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
    parser.add_argument('--n_data', type=int, default=1000,
                        help='test batch size for inference')
    parser.add_argument('--loss_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--quant_method', type=str, default='hqq',
                        help='')
    parser.add_argument('--eval_ppl', action='store_true', default=False)

    cfgs = parser.parse_args()
    main(cfgs)

