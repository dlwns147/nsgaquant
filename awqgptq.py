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
from utils.func import init_accelerator, get_net_info, clean_up, process_dtype, get_hfmodel
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer

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
    dtype = process_dtype(args.dtype)

    arch = dict()
    arch['linear'] = {linear: [args.bits] * config['n_block'] for linear in config['linear']}
    accelerator.print(arch)

    accelerator.print(get_net_info(arch, config, args.group_size))

    model_id = f'{args.model_path}/{args.model_name}'

    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        # latency_table=latency_table,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        group_size=args.group_size,
        dtype=dtype,
        clip_asym=args.clip_asym
    )
    
    # linear_bits = np.concatenate(list(arch['linear'].values()))
    # do_owq = ((linear_bits - linear_bits.astype(int)).sum() != 0)
    # print(f'do_owq : {do_owq}, use_awq_or_gptq : {use_awq_or_gptq}')
    
    # if args.bits == 16:
    #     model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
    # elif use_awq_gptq_qeft:
    #     model = get_quantized_model(method, arch, model_id, device_map, dtype=dtype, config=config, group_size=args.group_size, prune='layer_prune' in args.method, do_owq='qeft' in args.method, owq_path=args.outlier_path, clip_asym=args.clip_asym)
    # else:
    model = evaluator.sample(arch)

    # del evaluator
    # clean_up()
    # from hqq.utils.patching import prepare_for_inference
    # prepare_for_inference(model, backend="bitblas") 
    # lat = measure_latency(model, True, 'cuda', batch_size=1)
    # print(f'lat : {lat}, token/s = {128/lat}')
    # exit()

    metric, complexity = evaluator.eval(arch=arch, metric='ppl', model=model, accelerator=accelerator)
    accelerator.print(arch)
    print(f'complexity: {[f"{k}: {v}" for k, v in complexity.items()]}')
    print(f'ppl: {[p for p in metric.values()]}\n')
    
    del evaluator
    clean_up()
    print(f'memory : {torch.cuda.memory_allocated()}')
    
    if args.zeroshot:        
        results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, num_fewshot=args.num_fewshot, batch_size=args.zeroshot_batch_size)
        acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
        acc = [task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
        em_strict = [task_result['exact_match,strict-match'] if 'exact_match,strict-match' in task_result else 0 for task_result in results.values()]
        em_flexible = [task_result['exact_match,flexible-extract'] if 'exact_match,flexible-extract' in task_result else 0 for task_result in results.values()]
        em = em_strict + em_flexible
        
        task = list(results.keys())
        avg_acc_norm = np.mean(acc_norm)
        avg_acc = np.mean(acc)
        print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        print(f'task : {task}')
        print(f'acc_norm : {acc_norm}')
        print(f'em : {em}')
        
    print(args)
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--dtype', type=str, default='auto', choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat', 'bf16', 'auto'],
                        help='')
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
    parser.add_argument('--method', type=str, nargs='+', default=[], choices=['fp16', 'awq', 'gptq', 'hqq', 'qeft'],
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
    parser.add_argument('--zeroshot_batch_size', type=int, default=None,
                        help='')
    parser.add_argument('--num_fewshot', type=int, default=None,
                        help='')

    cfgs = parser.parse_args()
    main(cfgs)
