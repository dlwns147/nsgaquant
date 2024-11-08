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


def gen_data_layer(args):
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    search_space = LlamaSearchSpace(
        num_blocks=config['n_block'],
        quant_model_bits=[16],
        config=config,
        pass_linear_list=args.pass_linear_list,
        sec_obj=args.sec_obj,
        sec_obj_range=args.sec_obj_range,
        layer_prune_range=args.layer_prune_range
    )
    
    evaluator = LlamaEvaluator(
        config=config,
        model_name=args.model_name,
        method=args.method,
        seqlen=args.seqlen,
        quant_model_bits=[16],
        n_sample=args.n_sample,
        datasets=[args.dataset],
        loss_func=args.loss_func
    )
    ppl_archive = list()
    loss_archive = list()

    # complexity_list = list()
    archs = search_space.initialize(args.n_data)
    for arch in tqdm(archs):
        iter_start = time()

        # ppl, complexity = evaluator.eval(arch, 'ppl')
        # ppl_archive.append([arch, ppl[args.dataset], complexity[args.sec_obj]])
        evaluator.sample(arch)
        loss, complexity = evaluator.eval(arch, metric='loss', loss_func=args.loss_func)
        loss = np.nan_to_num(loss[args.dataset], nan=args.max_value)
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
    gen_data_layer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--method', type=str, nargs='+', default=['layer_prune'],
                        help='')
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
    parser.add_argument('--sec_obj', type=str, default='sparsity',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[0.5, 1.], 
                        help='')
    parser.add_argument('--max_value', type=float, default=10,
                        help='')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--layer_prune_range', type=float, nargs='+', default=[0., 1.], 
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

