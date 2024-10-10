import argparse
from time import time
import csv
import torch
import numpy as np

import gc
import json

from evaluator import LlamaEvaluator
from utils.func_utils import get_net_info

def greedy_search_linear(args):

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
    
    alive_linear_list = []
    # replaced_linear_list = []
    n_block = config['n_block']
    
    loss_list = dict()
    ppl_list = list()
    arch = {l: [] for l in config['linear']}
    
    for blk_idx in range(n_block):
        for linear in config['linear']:
            key = f'{blk_idx}.{linear}'
            arch[linear].append(args.large_model_bits if args.descending else args.small_model_bits)
            alive_linear_list.append(key)

    min_loss_list = list()
    min_loss_linear_list = list()
    bits_list = list()
    iter_time_list = list()
    iter_time_ppl_list = list()

    # check start time
    start_point = time()

    phase = 0
    while True:
        cur_bit = get_net_info(arch, config)['bits']
        if (not args.descending and cur_bit >= args.target_bit) or (args.descending and cur_bit <= args.target_bit) or len(alive_linear_list) == 0:
            break
        phase += 1
        phase_start_point = time()

        min_loss = 1e99
        min_loss_blk_idx = -1
        min_loss_linear = None

        for blk_idx in range(n_block):
            for linear in config['linear']:
                key = f'{blk_idx}.{linear}'
                if key in alive_linear_list:

                    linear_start = time()
                    arch[linear][blk_idx] = args.small_model_bits if args.descending else args.large_model_bits
                    loss, _ = evaluator.eval(arch, metric='loss')
                    loss = loss[args.dataset]
                    loss_list[key] = loss

                    if loss < min_loss:
                        min_loss = loss
                        min_loss_blk_idx = blk_idx
                        min_loss_linear = linear

                    arch[linear][blk_idx] = args.large_model_bits if args.descending else args.small_model_bits

                    linear_time = time() - linear_start
                    print(f"Phase {phase}, current bit: {cur_bit:.2f}, [{blk_idx}.{linear} replaced] Loss={loss_list[key]:.3f}, Current Min Loss={min_loss:.3f} / Layer {min_loss_blk_idx}.{min_loss_linear}, time: {linear_time:.2f}s") 
        selected_layer = f'{min_loss_blk_idx}.{min_loss_linear}'
        alive_linear_list.remove(selected_layer)
        # replaced_linear_list.append(selected_layer)
        arch[min_loss_linear][min_loss_blk_idx] = args.small_model_bits if args.descending else args.large_model_bits

        cur_bit = get_net_info(arch, config)['bits']
        bits_list.append(cur_bit)
        min_loss_list.append(min_loss)
        min_loss_linear_list.append(selected_layer)

        phase_time_elapsed = time() - phase_start_point
        iter_time_list.append(phase_time_elapsed)

        # remove block causing the least snlls increase 
        print(f"Phase_time_elapsed (s): {phase_time_elapsed:.2f}s")
        print(f"[SELECTED linear: {selected_layer}, Loss={min_loss:.3f}, Bits: {cur_bit:.3f}") 

        if args.loss_csv_file:
            with open(args.loss_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(min_loss_linear_list)
                write.writerow(bits_list)
                write.writerow(min_loss_list)
                write.writerow(iter_time_list)

        if args.eval_ppl_iter:        
            ppl, _ = evaluator.eval(arch, metric='ppl')
            ppl_phase_time_elapsed = time() - phase_start_point
            iter_time_ppl_list.append(ppl_phase_time_elapsed)
            if args.ppl_csv_file:
                ppl_list.append(ppl[args.dataset])
                with open(args.ppl_csv_file, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(min_loss_linear_list)
                    write.writerow(bits_list)
                    write.writerow(ppl_list)
                    write.writerow(iter_time_ppl_list)

    finish_point = time()
    time_elapsed = finish_point - start_point

    print(f"Time_Elapsed: {time_elapsed}")
    print(args)

    # if args.eval_ppl:
    #     print(f"Starting PPL evaluation...")
    #     # model = block_remove(model, copy.deepcopy(removal_list))
    #     model.config.use_cache = use_cache

    #     w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
    #     print(f"WikiText-2 PPL = {w2_ppl:.2f}")

    #     c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
    #     print(f"C4 PPL = {c4_ppl:.2f}")

    # if args.eval_zeroshot:
    #     del model
        
    #     print(f"Starting Zero-shot tasks evaluation...")
    #     if '30b' or '66b' or '70b' in model_name:
    #         parallelize = True
    #     else:
    #         parallelize = False

    #     tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

    #     results = eval_zero_shot(model_name, skip_attn_list, skip_mlp_list, tasks, parallelize=parallelize)
    #     results = results['results']

def main(args):
    greedy_search_linear(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_method', type=str, default='',
                        help='')
    parser.add_argument('--large_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_bits', type=float, default=4,
                        help='test batch size for inference')
    parser.add_argument('--small_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--small_model_bits', type=float, default=2,
                        help='test batch size for inference')
    parser.add_argument('--target_bit', type=float, default=3,
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
    parser.add_argument('--eval_ppl_iter', action='store_true', default=False)
    parser.add_argument('--eval_ppl', action='store_true', default=False)
    parser.add_argument('--eval_zeroshot', action='store_true', default=False)
    parser.add_argument('--descending', action='store_true', default=False)

    cfgs = parser.parse_args()
    main(cfgs)

