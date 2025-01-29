import os
import json
import torch
import argparse
import numpy as np
from utils import get_correlation
from evaluator import LlamaEvaluator
from tqdm import tqdm
from time import time
from copy import deepcopy

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PolynomialMutation

from search_space.llama import LlamaMultiObjLayerSearchSpace
from predictor.factory import get_predictor
from utils.func import get_net_info, init_accelerator
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntPolynomialMutation, MyTwoPointCrossover, MyUniformCrossover, IntegerFromFloatMutation

class Search:
    def __init__(self, config, accelerator, device_map, kwargs):
        self.args = deepcopy(kwargs)
        self.config = config
        self.device_map = device_map

        self.save_path = kwargs.pop('save', 'save')  # path to save results
        self.result_file = kwargs.pop('result_file', 'results.txt')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        # self.sec_obj = kwargs.pop('sec_obj', 'bits')  # second objective to optimize simultaneously
        self.comp_obj = kwargs.pop('comp_obj', ['bits'])  # second objective to optimize simultaneously
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.predictor = kwargs.pop('predictor', 'mlp')  # which surrogate model to fit
        # self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        # self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.dataset = kwargs.pop('dataset', 'wikitext2')  # which dataset to run search on
        self.loss_func = kwargs.pop('loss_func', 'jsd')

        self.method = kwargs.pop('method', '')
        self.quant_model_paths = kwargs.pop('quant_model_paths', [])
        self.quant_model_bits = kwargs.pop('quant_model_bits', [])
        # self.sec_obj_range = kwargs.pop('sec_obj_range', [])
        self.comp_obj_min = kwargs.pop('comp_obj_min', [])
        self.comp_obj_max = kwargs.pop('comp_obj_max', [])
        # assert len(self.sec_obj_range) == 2, "len(sec_obj_range) should be 2"
        assert len(self.comp_obj) == len(self.comp_obj_min) and len(self.comp_obj_min) == len(self.comp_obj_max)
        # self.layer_prune_range = kwargs.pop('layer_prune_range', [1, 1])

        model_name = kwargs.pop('model_name', 'Llama-2-7b-hf')
        model_path = kwargs.pop('model_path', 'meta-llama')
        model_id=f'{model_path}/{model_name}'
        self.metric = kwargs.pop('metric', 'loss')
        outlier_path = kwargs.pop('outlier_path' , '')
        base_outlier_bits = sorted(kwargs.pop('base_outlier_bits', []))
        n_outlier = kwargs.pop('n_outlier' , 0)
        # JG add
        latency_table = kwargs.pop('latency_table_file', None)
        self.latency_table = None
        if latency_table is not None:
            with open(latency_table, 'r') as f:
                self.latency_table = json.load(f)
        # self.config['mpe_table_json'] = kwargs.pop('mpe_table_json', '/NAS/JG/QAS4SD/llama2_7b_lpe_24bit.json')

        assert (outlier_path and base_outlier_bits and n_outlier > 0) or (not outlier_path and not base_outlier_bits and n_outlier == 0), "must use outlier_path, outlier_bits and n_outlier together when using outlier channel"
        
        outlier_bits = {l: [] for l in config['linear']}
        if outlier_path and base_outlier_bits and n_outlier > 0:
            for linear in config['linear']:
                for base_bits in base_outlier_bits:
                    _, in_dim = config['linear_shape'][linear]
                    avg_linear_bits = ((in_dim - n_outlier) * base_bits + n_outlier * 16) / (in_dim)
                    outlier_bits[linear].append(avg_linear_bits)
        
        self.evaluator = LlamaEvaluator(
            self.config,
            accelerator=accelerator,
            model_id=model_id,
            method=self.method,
            quant_model_paths=self.quant_model_paths,
            quant_model_bits=self.quant_model_bits,
            outlier=torch.load(outlier_path) if outlier_path else None,
            seqlen=kwargs.pop('seqlen', 2048),
            n_sample=kwargs.pop('n_sample', 128),
            datasets=[kwargs.pop('dataset', 'wikitext2')],
            loss_func=self.loss_func,
            device_map=device_map,
            latency_table=self.latency_table
        )
        self.search_space = LlamaMultiObjLayerSearchSpace(
            n_block=self.config['n_block'],
            pass_layer_list=kwargs.pop('pass_layer_list', []),
            comp_obj=self.comp_obj,
            comp_obj_min=self.comp_obj_min,
            comp_obj_max=self.comp_obj_max,
            config=self.config,
            # layer_prune_range=self.layer_prune_range,
            latency_table=self.latency_table
        )
        self.ga_pop_size = kwargs.pop('ga_pop_size', 40)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        self.debug = kwargs.pop('debug', False)
        self.ga_algorithm = kwargs.pop('ga_algorithm', 'nsga2')
        self.max_value = kwargs.pop('max_value', 5)
        self.crossover_prob = kwargs.pop('crossover_prob', 0.9)
        self.mut_prob = kwargs.pop('mut_prob', 0.1)
        accelerator.wait_for_everyone()
        
    def search(self, accelerator):
        total_start = time()
        start_it = 1
        
        if self.resume:
            archive, start_it = self._resume_from_dir()

        else:
            # the following lines corresponding to Algo 1 line 1-7 in the paper
            archive = []

            # Design Of Experiment
            if accelerator.is_main_process:
                if self.iterations < 1:
                    arch_doe = self.search_space.sample(
                        n_samples=self.n_doe,
                        pool=[x[0] for x in archive])
                else:
                    arch_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])
            else:
                arch_doe = list()
            arch_doe = accelerator.gather_for_metrics(arch_doe, use_gather_object=True)
            accelerator.wait_for_everyone()

            # parallel evaluation of arch_doe
            metric, complexity = self._evaluate(archs=arch_doe, accelerator=accelerator)
            complexity = [[x[obj] for obj in self.comp_obj] for x in complexity]
            
            if accelerator.is_main_process:
                # store evaluated / trained architectures
                for a, m, c in zip(arch_doe, metric, complexity):
                    archive.append([a, m, *c])

        if accelerator.is_main_process:
            # reference point (nadir point) for calculating hypervolume
            comp_obj_ref_pt_list = [np.array([np.max([x[1] for x in archive]), np.max([x[2+i] for x in archive])]) for i in range(len(self.comp_obj))]
            # bits_ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])
            # lat_ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[3] for x in archive])])
            accelerator.print(f'data preparation time : {time() - total_start:.2f}s')
        accelerator.wait_for_everyone()

        # main loop of the search
        for it in range(start_it, self.iterations + 1):
            if accelerator.is_main_process:
                accelerator.print(self.args)
                iter_start = time()

                # construct accuracy predictor surrogate model from archive
                # Algo 1 line 9 / Fig. 3(a) in the paper
                predictor_start = time()
                metric_predictor, a_metric_pred = self._fit_acc_predictor(archive, device=accelerator.device)
                predictor_time = time() - predictor_start

                # search for the next set of candidates for high-fidelity evaluation (lower level)
                # Algo 1 line 10-11 / Fig. 3(b)-(d) in the paper
                next_start = time()
                candidates, c_metric_pred = self._next(archive, metric_predictor, self.n_iter)
                next_time = time() - next_start
            else:
                candidates = list()
            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates, use_gather_object=True)

            # high-fidelity evaluation (lower level)
            # Algo 1 line 13-14 / Fig. 3(e) in the paper
            c_metric, complexity = self._evaluate(archs=candidates, accelerator=accelerator)
            complexity = [[x[obj] for obj in self.comp_obj] for x in complexity]

            if accelerator.is_main_process:
                # check for accuracy predictor's performance
                rmse, rho, tau = get_correlation(
                    np.vstack((a_metric_pred, c_metric_pred)), np.array([x[1] for x in archive] + c_metric))

                # add to archive
                # Algo 1 line 15 / Fig. 3(e) in the paper
                    
                for a, m, c in zip(candidates, c_metric, complexity):
                    archive.append([a, m, *c])

                # calculate hypervolume
                # hv = self._calc_hv(
                #     ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))
                
                hv_list = [self._calc_hv(
                    comp_obj_ref_pt_list[i], np.column_stack(([x[1] for x in archive], [x[2+i] for x in archive]))) for i, obj in enumerate(self.comp_obj)]

                iter_time = time() - iter_start
                # print iteration-wise statistics
                accelerator.print(f"Iter {it}: hv = {hv_list}, iter time : {(time() - iter_start):.2f}s, predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                accelerator.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall’s Tau = {tau:.4f}")
                accelerator.print(f'iteration time : {iter_time:.2f}s')

                # dump the statistics
                os.makedirs(self.save_path, exist_ok=True)
                with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                    json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv_list,
                            'surrogate': {
                                'model': self.predictor, 'name': metric_predictor.name,
                                'winner': metric_predictor.winner if self.predictor == 'as' else metric_predictor.name,
                                'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time}, 'iteration' : it}, handle)
                if self.debug:
                    import matplotlib.pyplot as plt
                    n_obj = len(self.comp_obj)
                    fig, axes = plt.subplots(nrows=1, ncols=n_obj, figsize=(5 * n_obj, 5))
                    for i in range(n_obj):
                        comp = np.array([x[2+i] for x in archive])  # second obj. (complexity)
                        perf = np.array([x[1] for x in archive])  # performance
                        axes[i].scatter(comp, perf, s=5, facecolors='none', edgecolors='b', label='archive')
                        comp = np.array(complexity)[:, i]
                        perf = np.array(c_metric)
                        axes[i].scatter(comp, perf, s=10, color='r', label='candidates evaluated')
                        comp = np.array(complexity)[:, i]
                        perf = c_metric_pred[:, 0]
                        axes[i].scatter(comp, perf, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
                        axes[i].legend(loc="upper right")
                        axes[i].set_xlabel(f'f{2+i}')
                        axes[i].grid(c='0.8') 
                    axes[0].set_ylabel('f1')
                    fig.tight_layout() 
                    plt.savefig(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            total_time_elapsed = time() - total_start
            accelerator.print(f'total time elapsed : {total_time_elapsed:.2f}s')

            sentences = []
            for k, v in self.args.items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'Total time: {total_time_elapsed:.2f}s')
            # sentences.append("\n")

            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

            accelerator.print(self.args)
        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """

        with open(self.resume, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive'] + resume_file['candidates']
            it = resume_file['iteration']

        return archive, it + 1

    def _evaluate(self, archs, accelerator):
        # metric_list, complexity_list = [], []
        metric_list, complexity_list = [], []
        for arch in tqdm(archs, desc='Eval Arch'):
            metric, complexity = self.evaluator.eval(accelerator=accelerator, arch=arch, metric=self.metric, loss_func=self.loss_func)
            metric_list.append(np.nan_to_num(metric[self.dataset], nan=self.max_value))
            complexity_list.append(complexity)
            # bits_list.append(complexity['bits'])
            # lat_list.append(complexity['latency'])

        return metric_list, complexity_list

    def _fit_acc_predictor(self, archive, device='cpu'):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        # inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        # assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        metric_predictor = get_predictor(self.predictor, inputs, targets, device=device)

        return metric_predictor, metric_predictor.predict(inputs)
    
    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        # F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        F = np.column_stack([[x[1+i] for x in archive] for i in range(len(self.comp_obj) + 1)])
        # import pdb; pdb.set_trace()
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initiate a multi-objective solver to optimize the problem
        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,  # initialize with current nd archs
            # crossover=TwoPointCrossover(prob=0.9),
            crossover=BinomialCrossover(prob=0.9, n_offsprings=1),
            # crossover=BinomialCrossover(prob=0.9, n_offsprings=2),
            # crossover=MyTwoPointCrossover(prob=0.9, n_offsprings=1),
            # mutation=IntPolynomialMutation(eta=1.0),
            mutation=IntegerFromFloatMutation(clazz=PolynomialMutation, eta=1.0, prob=self.mut_prob),
            # mutation=PolynomialMutation(prob=self.mut_prob, eta=1.0),
            # mutation=IntPolynomialMutation(prob=self.mut_prob, eta=1.0),
            eliminate_duplicates=True)
        
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self.search_space, predictor, self.config, self.latency_table)
        
        # kick-off the search
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        # indices = self._subset_selection(res.pop[not_duplicate], F[front, 1:], K, self.subset_pop_size)
        # pop = res.pop[not_duplicate][indices]
        pop = res.pop[not_duplicate]


        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        # return candidates, predictor.predict(np.stack([self.search_space.encode_predictor(self.search_space.decode(x)) for x in pop.get("X")]))
        return candidates, predictor.predict(pop.get("X"))

    # @staticmethod
    def _subset_selection(self, pop, nd_F, K, pop_size):
        # candidates = np.array([get_net_info(self.search_space.decode(x), self.config, self.latency_table)[self.sec_obj] for x in pop.get("X")])
        # problem = SubsetProblem(candidates, nd_F, K)
        # import pdb; pdb.set_trace()
        problem = SubsetProblem(pop.get("F")[:, 1:], nd_F, K)
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self, search_space, predictor, config, latency_table):
        n_block, n_layer = search_space.n_block, search_space.n_layer
        n_comp_obj = len(search_space.comp_obj)
        super().__init__(n_var=n_block * n_layer, n_obj=1 + n_comp_obj, n_constr=2 * n_comp_obj, type_var=int)

        self.ss = search_space
        self.predictor = predictor
        self.xl = np.zeros((n_block, n_layer))
        self.xu = np.ones((n_block, n_layer))
        
        self.config = config
        self.latency_table = latency_table

        for pass_layer in self.ss.pass_layer_list:
            blk, layer = pass_layer.split('.', 1)
            blk = int(blk)
            
            layer_idx = config['layer'].index(layer)
            self.xl[blk, -n_layer + layer_idx] = 1
        
        self.xl = self.xl.flatten()
        self.xu = self.xu.flatten()


    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        # metrics = self.predictor.predict(np.stack([self.ss.encode_predictor(self.ss.decode(_x)) for _x in x]))[:, 0]
        metrics = self.predictor.predict(x)[:, 0]

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            arch = self.ss.decode(_x)
            info = get_net_info(arch, self.config, self.latency_table)
            f[i, 0] = metric
            # f[i, 1] = info[self.ss.sec_obj]
            for j in range(len(self.ss.comp_obj)):
                f[i, 1 + j] = info[self.ss.comp_obj[j]]
                g[i, 2 * j] = 1 - info[self.ss.comp_obj[j]] / self.ss.comp_obj_min[j]
                g[i, 2 * j + 1] = info[self.ss.comp_obj[j]] / self.ss.comp_obj_max[j] - 1

            # g[i, 0] = 1 - info[self.ss.sec_obj] / self.ss.sec_obj_range[0]
            # g[i, 1] = info[self.ss.sec_obj] / self.ss.sec_obj_range[1] - 1
            # g[i, 2 * (len(self.ss.comp_obj))] = 1 - info['sparsity'] / self.ss.layer_prune_range[0]
            # g[i, 2 * (len(self.ss.comp_obj)) + 1] = info['sparsity'] / self.ss.layer_prune_range[1] - 1

        out["F"] = f
        out["G"] = g

class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        # f = np.full((x.shape[0], self.n_obj), np.nan)
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            # for j in range(self.n_obj):
            #     tmp = np.sort(np.concatenate((self.archive[:, j], self.candidates[_x][:, j])))
            #     f[i, j] = np.std(np.diff(tmp))
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])), axis=0)
            f[i, 0] = np.std(np.diff(tmp, axis=0))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = Search(config=config, accelerator=accelerator, device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['bits'], 
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=50,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    # parser.add_argument('--n_gpu', type=int, default=1,
    #                     help='number of gpus per process')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for selecting calibration set, etc.')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='sample number of the calibration set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss)')
    parser.add_argument('--pass_linear_list', type=str, nargs='+', default=[], 
                        help='linear list not to replace')
    parser.add_argument('--pass_layer_list', type=str, nargs='+', default=[], 
                        help='linear list not to replace')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='config file to read the model meta data')
    parser.add_argument('--ga_pop_size', type=int, default=40,
                        help='population size of the NSGA stage')
    parser.add_argument('--subset_pop_size', type=int, default=100,
                        help='population size of the subset selection stage')
    parser.add_argument('--debug', action='store_true', help='visualization of each iteration results')
    # parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
    #                     help='')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[2], 
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[4], 
                        help='')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--ga_algorithm', type=str, default='nsga2',
                        help='')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--max_value', type=float, default=50,
                        help='')
    parser.add_argument('--crossover_prob', type=float, default=0.9,
                        help='')
    parser.add_argument('--mut_prob', type=float, default=0.05,
                        help='')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--layer_prune_range', type=float, nargs='+', default=[1, 1], 
                        help='')
    parser.add_argument('--use_linear_group', action='store_true', help='')
    parser.add_argument('--base_outlier_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--n_outlier', type=int, default=0, 
                        help='')
    parser.add_argument('--only_outlier_bits', action='store_true', help='')
    parser.add_argument('--latency_table_file', type=str, default='',
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

