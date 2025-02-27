"""
알고리즘이랑 병합한 버전
"""

import os
import gc
import json
import argparse
import torch
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder

from evaluator import LlamaEvaluator
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer

class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        # if self.normalize:
        #     F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

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

            np.warnings.filterwarnings('ignore')
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

    latency_table = None
    if args.latency_table_file:
        with open(args.latency_table_file, 'r') as f:
            latency_table = json.load(f)
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    # preferences
    if args.prefer:
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

    with open(args.expr, 'r') as f:
        result_json = json.load(open(args.expr))
        archive = result_json['archive'] + result_json['candidates']
    # subnets, metric, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    subnets, metric = [v[0] for v in archive], [v[1] for v in archive]
    sec_obj = [get_net_info(n, config, latency_table)[args.sec_obj] for n in subnets]
    # sec_objs = [[get_net_info(n, config, latency_table)[o] for n in subnets] for o in args.sec_obj]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    # F = np.column_stack((metric, *sec_objs))[sort_idx, :]

    if len(args.sec_obj_range) % 2 == 0:
        # assert args.sec_obj_range[0] >= min(args.quant_model_bits) and args.sec_obj_range[1] <= max(args.quant_model_bits), f'Target bits range should be in [small model bits, large model bits]'
        range_idx = np.argwhere(np.logical_and(F[:, 1] > args.sec_obj_range[0], F[:, 1] < args.sec_obj_range[1])).flatten()
        pf = F[range_idx, :]
        ps = np.array(subnets)[sort_idx][range_idx]
        # import code; code.interact("check", local = dict(globals(), **locals()))

    elif args.only_front:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pf = F[front, :]
        ps = np.array(subnets)[sort_idx][front]
        
    else:
        pf = F
        ps = np.array(subnets)[sort_idx]

    min_num_2bits = 100000
    min_num_2ibts_idx = 0
    min_num_2bits_ratio = 100000
    min_num_2bits_ratio_idx = 0

    # for p in ps:
    if len(ps) == 0:
        print(f'length of ps : {len(ps)}')    
        return
    for i, p in enumerate(ps):
        net_info_p = get_net_info(p, config, latency_table)
        num_2bits = net_info_p['2bits']
        num_2bits_ratio = net_info_p['2bits_ratio']

        if num_2bits < min_num_2bits:
            min_num_2bits = num_2bits
            min_num_2bits_idx = i

        if num_2bits_ratio < min_num_2bits_ratio:
            min_num_2bits_ratio = num_2bits_ratio
            min_num_2bits_ratio_idx = i

        I = ASF().do(pf, weights).argsort()[:args.n]

    with open('benchmark/2bit.txt', 'a') as f:
        f.write(f'length of ps : {len(ps)}, bits : {args.sec_obj_range}\n')
        f.write(f'min_num_2bits : {min_num_2bits}, min_num_2bits_idx : {min_num_2bits_idx}\n')
        f.write(f'min_num_2bits_ratio : {min_num_2bits_ratio}, min_num_2bits_ratio_idx : {min_num_2bits_ratio_idx}\n')
        f.write(f'choosed idx : {I}\n')
        f.write(f'arch : {ps[I]}\n')

        return
        
    if args.prefer:
        # choose the architectures thats closest to the preferences
        I = ASF().do(pf, weights).argsort()[:args.n]
    # else:
    #     # choose the architectures with highest trade-off
    #     dm = HighTradeoffPoints(n_survive=args.n)
    #     I = dm.do(pf)
    else:
        I = range(len(pf))
    # always add most accurate architectures
    # I = np.append(I, 0)

    # with open('/NAS/Woo/Automation/autoopt/archs/post_search/7b_owq/results_arch.json', 'r') as f:
    #     data = json.load(f)
    #     archs = data['archive']

    # for idx in I:
        # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}')
        # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, attns : {int(sum(ps[idx]["layer"]["self_attn"]))}, mlps  : {int(sum(ps[idx]["layer"]["mlp"]))}')
        
        # archs.append([ps[idx], pf[idx, 1]])
        
    # with open('/NAS/Woo/Automation/autoopt/archs/post_search/7b_owq/results_arch.json', 'w') as f:
    #     json.dump({'archive': archs}, f, ensure_ascii=False, indent=4)

    latency_table = None
    if args.latency_table_file:
        with open(args.latency_table_file, 'r') as f:
            latency_table = json.load(f)

    model_id = f'{args.model_path}/{args.model_name}'
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        latency_table=latency_table
    )

    ## customizing
    from autoquant.autoquant.algorithm import get_quantized_model
    del evaluator.model
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = get_tokenizer(model_id)
    global field

    # ppl_list = {dataset: [] for dataset in args.datasets}
    arch_list = []
    ppl_list = []
    bits_list = []
    param_list = []
    sparsity_list = []
    metric_list = []
    latency_list = []
    complexity_list = []
    for idx in tqdm(I):
        # idx = 4
        # print(f'idx : {idx}')
        # arch = ps[idx]
        # # model = evaluator.sample(arch)
        # # del evaluator.quant_models
        # # import gc
        # # gc.collect()
        # # torch.cuda.empty_cache()
        # # print(f'memory : {torch.cuda.memory_allocated()}')
        # # exit()
        # arch = {'linear': {l: [3] * config['n_block'] for l in config['linear']}, 'layer': {l: [1] * config['n_block'] for l in config['layer']}}
        # print(f'arch : {arch}')

        arch = ps[idx]

        # arch['layer']['self_attn'][0] = 0
        # arch['layer']['mlp'][0] = 0

        result = {}
        method = get_quantized_model(args.method[0], arch, model_id, 'cuda:0', do_prune = args.do_prune, do_owq = args.do_owq, owq_path = torch.load(args.outlier_path) if args.do_owq else None)
        model = method.model
        model = model.to('cuda:0')

        metric, complexity = evaluator.eval_woo(arch=arch, model = model, metric='ppl', accelerator=accelerator)

        # metric, complexity = evaluator.eval(arch=arch, metric='ppl', accelerator=accelerator)
        # model = evaluator.sample(arch)
        # latency = measure_latency(evaluator.sample(arch), generation=True, device=model.device) if args.latency else 0
        latency = 0

        arch_list.append(arch)
        metric_list.append(pf[idx, 0])
        ppl_list.append({d: metric[d] for d in args.datasets})
        bits_list.append(complexity['bits'])
        param_list.append(complexity['params'])
        sparsity_list.append(complexity['sparsity'])
        complexity_list.append(complexity[args.sec_obj])  
        latency_list.append(latency)
        print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, ppl: {[p for p in metric.values()]}, metric: {pf[idx, 0]:.4f} complexity: {complexity}, latency: {latency}\n')

        result['bits'] = complexity['bits']
        result['params'] = complexity['params']
        result['sparsity'] = complexity['sparsity']
        result['wikitext2'] = metric['wikitext2']
        result['c4'] = metric['c4']
        
        if args.zeroshot:
            # results = eval_zeroshot(evaluator.sample(arch), tokenizer=get_tokenizer(model_id), batch_size=args.zeroshot_batch_size)
            # results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), batch_size='auto')
            results = eval_zeroshot(model, tokenizer=tokenizer, batch_size='auto')
            avg_acc = np.mean([task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()])
            print(f'avg_acc : {avg_acc}, results : {results}')
            for task, task_result in results.items():
                if 'acc_norm,none' in task_result:
                    print(f'{task} acc_norm : {task_result["acc_norm,none"]}')

                    result[task] = task_result['acc_norm,none']
                else:
                    print(f'{task} acc : {task_result["acc,none"]}')

                    result[task] = task_result['acc,none']

            result['avg'] = avg_acc

        with open(args.output_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=field)
            writer.writerow(result)

            # row_list = []
            # for task_name, task_result in results:
            #     row = [task_name]
            #     head_list = ['head']
            #     for head, metric in task_result:
            #         row.append(metric)
            #         head_list.append(head)
            #     row_list.append(row)
            # with open(args.zeroshot_csv_file, 'r') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(head_list)
            #     for row in row_list:
            #         writer.writerow(row)

        print(args)
        # exit()
        return
    # exit()
    return

    if args.debug:
        # print(ps[I])
        # plot = Scatter()
        # plot.add(pf, alpha=0.2)
        # plot.add(pf[I, :], color="blue", s=10)
        # plot.add(gs_data, color="red", s=10)
        # plot.show()
        # plot.save(os.path.join(args.save, "best_trade_off_line.png"))
        os.makedirs(args.save, exist_ok=True)
        
        plt.scatter(complexity_list, [p[args.datasets[0]] for p in ppl_list], color='b', s=5, label='NSGA2')
        if args.greedy_search_result_path:
            with open(args.greedy_search_result_path, 'r') as f:
                gs_data = list(csv.reader(f))
                gs_bits = list(map(float, gs_data[1]))[:-3]
                gs_metric = list(map(float, gs_data[2]))[:-3]
                plt.scatter(gs_bits, gs_metric, color='r', s=5, label='Greedy Search')
        
        plt.xlabel(f'{args.sec_obj}')
        plt.ylabel('PPL')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(args.save, "best_trade_off_line.png"), dpi=300)

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
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--greedy_search_result_path', type=str, default='',
                        help='')
    parser.add_argument('--last_layer', type=str, default='',
                        help='')
    parser.add_argument('--only_front', action='store_true', help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--latency_table_file', type=str, default=None,
                        help='')
    parser.add_argument('--latency', action='store_true', help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--zeroshot_csv_file', type=str, default=None,
                        help='')
    parser.add_argument('--zeroshot_batch_size', type=int, default=64,
                        help='')
    
    parser.add_argument('--output_path', type=str, default='',
                        help='')
    parser.add_argument('--do_prune', action='store_true', help='Whether to use pruning')
    parser.add_argument('--do_owq', action='store_true', help='Whether to use owq')

    cfgs = parser.parse_args()

    ## customizing
    global field
    field = ['bits', 'params', 'sparsity', 'wikitext2', 'c4', 'piqa', 'winogrande', 'hellaswag', 'arc_challenge', 'arc_easy', 'avg']

    with open(cfgs.output_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=field)
        writer.writeheader()


    # target_bit = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
    # target_bit = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
    # target_bit = [2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4.0]
    target_bit = [2.25, 2.75, 3.25]
    THRESHOLD = 0.005
    for i in target_bit:
        cfgs.sec_obj_range[0] = i - THRESHOLD
        cfgs.sec_obj_range[1] = i + THRESHOLD

        main(cfgs)
