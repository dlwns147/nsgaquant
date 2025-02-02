import argparse
import time
import csv

import numpy as np
import torch
import json
import csv
import time

from evaluator import LlamaEvaluator
from utils.func import init_accelerator


def layer_sensitivity(args):
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        model_id=f'{args.model_path}/{args.model_name}',
        method=args.method,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        loss_func=args.loss_func,
        datasets=[args.dataset],
        device_map=device_map,
        # dtype=torch.bfloat16
    )
    
    n_block = int(config['n_block'])
    n_linear = int(config['n_linear'])

    ppl = 0
    loss_list = dict()
    ppl_list = dict()
    # arch = {'linear': {l: [16] * n_block for lg in config['linear'] for l in lg.split(',')}, 'layer': {l: [1]* n_block for l in config['layer']}}
    arch = {'layer': {l: [1]* n_block for l in config['layer']}}
    
    for layer in config['layer']:
        for block_idx in range(n_block):
            ppl_list[f'{block_idx}.{layer}'] = 0
    accelerator.print(f'arch : {arch}')

    # import pdb; pdb.set_trace()
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
            accelerator.print(f'save to {args.loss_csv_file}')
        if args.eval_ppl and args.ppl_csv_file:
            with open(args.ppl_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(ppl_list.keys()))
                write.writerow(list(ppl_list.values()))
            accelerator.print(f'save to {args.ppl_csv_file}')

    finish_point = time.time()
    time_elapsed = finish_point - start_point

    accelerator.print(f"Time_Elapsed: {time_elapsed}")
    accelerator.print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    # parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
    #                     help='')
    # parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
    #                     help='')
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


    # # # greedy_result_path = '/NAS/SJ/sleb/csv/Llama-2-7b-hf_ppl_512_js.csv'
    # greedy_result_path = '/NAS/SJ/sleb/csv/Llama-2-13b-hf_ppl_512_js.csv'
    # with open(greedy_result_path, 'r') as f:
    #     greedy_result = list(csv.reader(f))
    #     selected_layer_list = greedy_result[0]

    # params_list = []
    # sparsity_list = []
    # for i, layer in enumerate(selected_layer_list):
    #     blk_idx, layer = layer.split('.')
    #     arch['layer'][layer][int(blk_idx)] = 0
    #     complexity = get_net_info(arch, config)
    #     print(f"{i + 1} {blk_idx}.{layer} | params : {complexity['params']:.3f}, sparsity : {complexity['sparsity']:.3f}")
    #     params_list.append(complexity['params'])
    #     sparsity_list.append(complexity['sparsity'])
    # greedy_result.append(params_list)
    # greedy_result.append(sparsity_list)

    # with open(greedy_result_path, 'w') as f:
    #     write = csv.writer(f)
    #     for r in greedy_result:
    #         write.writerow(r)
    
    # exit()

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")

    # for linear in config['linear']:
    #     for block_idx in range(n_block):
    #         arch['linear'][linear][block_idx] = min(args.quant_model_bits)

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")

    # exit()