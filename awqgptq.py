import os
import json
import argparse
import torch
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluator import LlamaEvaluator
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer
from quant.model import get_quantized_model
import gc

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True 

class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, normalize=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected
        self.normalize = normalize

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, estimate_bounds_if_none=True)
            # F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            # np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):
    print(args)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    latency_table = None

    model_id = f'{args.model_path}/{args.model_name}'

    use_awq_or_gptq = 'awq' in args.method or 'gptq' in args.method
    method = 'awq' if 'awq' in args.method else 'gptq' if 'gptq' in args.method else None
    
    if use_awq_or_gptq:
        args.quant_model_bits = []
        args.quant_model_paths = []

    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        latency_table=latency_table
    )

    arch = dict()
    arch['linear'] = {linear: [args.bits] * config['n_block'] for linear in config['linear']}
    accelerator.print(arch)
    
    linear_bits = np.concatenate(list(arch['linear'].values()))
    do_owq = ((linear_bits - linear_bits.astype(int)).sum() != 0)
    print(f'do_owq : {do_owq}, use_awq_or_gptq : {use_awq_or_gptq}')
    if args.bits == 16:
        from utils.func import get_hfmodel
        model = get_hfmodel(model_id, dtype='auto', device_map=device_map)
    elif use_awq_or_gptq:
        model = get_quantized_model(method, arch, model_id, device_map, config=config, group_size=args.group_size, prune='layer_prune' in args.method, do_owq=do_owq, owq_path=args.outlier_path, clip_asym=args.clip_asym)
    else:
        model = evaluator.sample(arch)
    metric, complexity = evaluator.eval(arch=arch, metric='ppl', model=model, accelerator=accelerator)
    accelerator.print(arch)
    print(f'ppl: {[p for p in metric.values()]}\n')
    
    if args.zeroshot:
        torch.cuda.empty_cache()
        gc.collect()
        
        results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), batch_size=args.zeroshot_batch_size, task_list=args.tasks)
        acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
        acc = [task_result['acc,none'] for task_result in results.values()]
        
        task = list(results.keys())
        avg_acc_norm = np.mean(acc_norm)
        avg_acc = np.mean(acc)
        print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        print(f'task : {task}')
        print(f'acc_norm : {acc_norm}')
        print(f'acc : {acc}')
    if use_awq_or_gptq:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(args)
    exit()

    sentences = []
    for k, v in vars(args).items():
        sentences.append(f"{k}: {v}\n")
    sentences.append("\n")
    for a, c, p in zip(arch_list, complexity_list, ppl_list):
        sentences.append(f"arch: {a}, bits: {c:.4f}, ppl: {p}\n")

    with open(os.path.join(args.save, args.results_file), 'w') as f:
        for sentence in sentences:
            f.write(sentence)

    with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'bits', 'params', 'sparsity', 'metric', 'latency'] + args.datasets)
        for a, b, p, s, m, l, ppl in zip(arch_list, bits_list, param_list, sparsity_list, metric_list, latency_list, ppl_list):
            writer.writerow([a, b, p, s, m, l] + list(ppl.values()))

    with open(os.path.join(args.save, args.results_arch_file), 'w') as f:
        json.dump({'archive': [[a, c, p] for a, c, p in zip(arch_list, complexity_list, ppl_list)]}, f, ensure_ascii=False, indent=4)

    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--bits', type=int, default=2,
                        help='')
    parser.add_argument('--group_size', type=int, default=128,
                        help='')
    parser.add_argument('--clip_asym', action='store_true', help='')
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['bits'], 
                        help='second objective to optimize simultaneously')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    # parser.add_argument('--expr', type=str, default='',
    #                     help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    # parser.add_argument('--high_tradeoff', action='store_true', help='')
    parser.add_argument('--high_tradeoff', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='')
    # parser.add_argument('--debug', action='store_true', help='')
    # parser.add_argument('--sec_obj', type=str, default='bits',
    #                     help='second objective to optimize simultaneously')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    # parser.add_argument('--greedy_search_result_path', type=str, default='',
    #                     help='')
    # parser.add_argument('--last_layer', type=str, default='',
    #                     help='')
    # parser.add_argument('--only_front', action='store_true', help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    # parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[],
    #                     help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    # parser.add_argument('--latency_table_file', type=str, default=None,
    #                     help='')
    parser.add_argument('--latency', action='store_true', help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['piqa','winogrande','hellaswag','arc_challenge','arc_easy', 'lambada_openai', 'boolq'])
    parser.add_argument('--zeroshot_csv_file', type=str, default=None,
                        help='')
    parser.add_argument('--zeroshot_batch_size', type=int, default=64,
                        help='')

    cfgs = parser.parse_args()
    main(cfgs)
