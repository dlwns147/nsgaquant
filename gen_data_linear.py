import torch
import argparse
from tqdm import tqdm
import numpy as np
import gc
import json
import torch.nn as nn

from time import time

from search_space.llama import LlamaSearchSpace
from evaluator import LlamaEvaluator


def gen_data_linear(args):
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    search_space = LlamaSearchSpace(
        num_blocks=config['n_block'],
        small_model_bits=args.small_model_bits,
        large_model_bits=args.large_model_bits,
        config=config,
        pass_linear_list=args.pass_linear_list,
        sec_obj_range=args.sec_obj_range
    )
    
    evaluator = LlamaEvaluator(
        config=config,
        model_name=args.model_name,
        quant_method=args.quant_method,
        large_model_path=args.large_model_path,
        large_model_bits=args.large_model_bits,
        small_model_path=args.small_model_path,
        small_model_bits=args.small_model_bits,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset]
    )
    ppl_archive = list()
    loss_archive = list()

    # complexity_list = list()
    archs = search_space.initialize(args.n_data)
    for arch in tqdm(archs):
        iter_start = time()

        # ppl, complexity = evaluator.eval(arch, 'ppl')
        # ppl_archive.append([arch, ppl[args.dataset], complexity[args.sec_obj]])
        loss, complexity = evaluator.eval(arch, 'loss')
        loss = np.nan_to_num(loss[args.dataset], nan=args.nan_value)
        loss_archive.append([arch, loss, complexity[args.sec_obj]])

        iter_time = time() - iter_start
        # print(f'complexity: {complexity:.3f}, loss : {loss:2f}, time : {iter_time:.2f}')
        print(f'{args.sec_obj}: {complexity[args.sec_obj]:.3f}, loss : {loss:2f}, time : {iter_time:.2f}s')
        # print(f'{args.sec_obj}: {complexity[args.sec_obj]:.3f}, ppl : {ppl[args.dataset]:.2f}, loss : {loss[args.dataset]:.2f}, time : {iter_time:.2f}s')
        # complexity_list.append(complexity)

        # if args.ppl_json_file:
        #     with open(args.ppl_json_file, 'w') as f:
        #         json.dump({'archive': ppl_archive}, f, ensure_ascii=False, indent=4)

        # if args.loss_json_file:
        #     with open(args.loss_json_file, 'w') as f:
        #         json.dump({'archive': loss_archive}, f, ensure_ascii=False, indent=4)
    # from matplotlib import pyplot as plt
    # plt.hist(complexity_list)
    # plt.show()
    # plt.savefig('test.png', dpi=300)


def main(args):
    gen_data_linear(args)


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
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--metric', type=str, default='ppl',
                        help='which accuracy predictor model to fit (ppl/loss)')
    parser.add_argument('--pass_linear_list', type=str, nargs='+', default=[], 
                        help='which accuracy predictor model to fit (ppl/loss)')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--n_data', type=int, default=1000,
                        help='test batch size for inference')
    parser.add_argument('--loss_json_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_json_file', type=str, default='',
                        help='')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--nan_value', type=float, default=50,
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

