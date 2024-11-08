import os
import json
import argparse
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

    # preferences
    if args.prefer:
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

    archive = json.load(open(args.expr))['archive']
    subnets, metric, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, sec_obj))[sort_idx, :]
    if args.only_front:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pf = F[front, :]
        ps = np.array(subnets)[sort_idx][front]
    elif len(args.target_obj_range) == 2:
        assert args.target_obj_range[0] >= 0 and args.target_obj_range[1] <= 1, f'Target sparsity range should be in [0, 1]'
        range_idx = np.argwhere(np.logical_and(F[:, 1] > args.target_obj_range[0], F[:, 1] < args.target_obj_range[1])).flatten()
        pf = F[range_idx, :]
        ps = np.array(subnets)[sort_idx][range_idx]

    else:
        pf = F
        ps = np.array(subnets)[sort_idx]
        
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

    arch = {'self_attn' : np.zeros((32)), 'mlp' : np.zeros((32))}
    for idx in I:
        # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}, removed_attn_layers: {np.where(np.array(ps[idx]["self_attn"]) == 0)}, removed_mlp_layers: {np.where(np.array(ps[idx]["mlp"]) == 0)}')
        # print(f"attn difference: {np.where(np.logical_xor(ps[idx]['self_attn'], arch['self_attn']))}, mlp difference: {np.where(np.logical_xor(ps[idx]['mlp'], arch['mlp']))}")
        print(f"{idx} attn: {np.where(np.array(ps[idx]['self_attn']) == 0)[0].tolist()}, mlp: {np.where(np.array(ps[idx]['mlp']) == 0)[0].tolist()}")
        arch = ps[idx]
        
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    evaluator = LlamaEvaluator(
        config,
        model_name=args.model_name,
        method=args.method,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets
    )

    # ppl_list = {dataset: [] for dataset in args.datasets}
    arch_list = []
    ppl_list = []
    complexity_list = []
    metric_list = []
    for idx in tqdm(I):
        arch = ps[idx]
        metric, complexity = evaluator.eval(arch, 'ppl', args.method)
        arch_list.append(arch)
        metric_list.append(pf[idx, 0])
        ppl_list.append({d: metric[d] for d in args.datasets})
        complexity_list.append(complexity[args.sec_obj])
        print(f'Selected arch[{idx}] bits: {pf[idx, 1]:.4f}, ppl: {[p for p in metric.values()]}, metric: {pf[idx, 0]:.4f}\n')

    if args.debug:
        # print(ps[I])
        # plot = Scatter()
        # plot.add(pf, alpha=0.2)
        # plot.add(pf[I, :], color="blue", s=10)
        # plot.add(gs_data, color="red", s=10)
        # plot.show()
        # plot.save(os.path.join(args.save, "best_trade_off_line.png"))
        os.makedirs(args.save, exist_ok=True)
        
        plt.scatter(complexity_list, [p['wikitext2'] for p in ppl_list], color='b', s=5, label='NSGA2')
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
        writer.writerow(['arch', args.sec_obj, 'metric'] + args.datasets)
        for a, c, m, p in zip(arch_list, complexity_list, metric_list, ppl_list):
            writer.writerow([a, c, m] + list(p.values()))

    with open(os.path.join(args.save, args.results_arch_file), 'w') as f:
        json.dump({'archive': [[a, c, p] for a, c, p in zip(arch_list, complexity_list, ppl_list)]}, f, ensure_ascii=False, indent=4)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='test batch size for inference')
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
    parser.add_argument('--debug', type=bool, default=True,
                        help='')
    parser.add_argument('--sec_obj', type=str, default='sparsity',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--datasets', type=str, nargs='+', default=['wikitext2'], 
                        help='linear list not to replace')
    parser.add_argument('--greedy_search_result_path', type=str, default='',
                        help='')
    parser.add_argument('--only_front', type=bool, default=True,
                        help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    parser.add_argument('--target_obj_range', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--method', type=str, default='layer_prune',
                        help='')

    cfgs = parser.parse_args()
    main(cfgs)
