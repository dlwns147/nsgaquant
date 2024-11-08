import os
import json
import torch
import shutil
import argparse
import subprocess
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

from search_space.llama import LlamaSearchSpace
from acc_predictor.factory import get_acc_predictor
from utils.func_utils import prepare_eval_folder, get_net_info
from utils.ga_utils import MySampling, BinaryCrossover, MyMutation, IntPolynomialMutation, MyTwoPointCrossover, MyUniformCrossover


class Search:
    def __init__(self, kwargs):
        self.args = deepcopy(kwargs)
        self.save_path = kwargs.pop('save', 'save')  # path to save results
        self.result_file = kwargs.pop('result_file', 'results.txt')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.sec_obj = kwargs.pop('sec_obj', 'bits')  # second objective to optimize simultaneously
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.predictor = kwargs.pop('predictor', 'mlp')  # which surrogate model to fit
        # self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        # self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.dataset = kwargs.pop('dataset', 'wikitext2')  # which dataset to run search on
        self.latency = self.sec_obj if "cpu" in self.sec_obj or "gpu" in self.sec_obj else None
        self.large_model_bits = kwargs.pop('large_model_bits', 4)
        self.small_model_bits = kwargs.pop('small_model_bits', 2)
        self.sec_obj_range = kwargs.pop('sec_obj_range', [self.small_model_bits, self.large_model_bits])
        assert len(self.sec_obj_range) == 2 and self.sec_obj_range[0] >= self.small_model_bits and self.sec_obj_range[1] <= self.large_model_bits, "len(sec_obj_range) should be 2, and sec_obj_range should be in (small_model_bits, large_model_bits)"
        
        model_name = kwargs.pop('model_name', 'meta-llama/Llama-2-7b-hf')
        self.metric = kwargs.pop('metric', 'ppl')
        self.predictor_data = kwargs.pop('predictor_data', '')
        with open(kwargs.pop('config', 'config/llama.json'), 'r') as f:
            self.config = json.load(f)[model_name]
        self.evaluator = LlamaEvaluator(
            self.config,
            model_name=model_name,
            quant_method=kwargs.pop('quant_method', ''),
            large_model_path=kwargs.pop('large_model_path', ''),
            large_model_bits=self.large_model_bits,
            small_model_path=kwargs.pop('small_model_path', ''),
            small_model_bits=self.small_model_bits,
            seqlen=kwargs.pop('seqlen', 2048),
            n_sample=kwargs.pop('n_sample', 128),
            datasets=[kwargs.pop('dataset', 'wikitext2')]
        )
        self.search_space = LlamaSearchSpace(
            num_blocks=self.config['n_block'],
            large_model_bits=self.large_model_bits,
            small_model_bits=self.small_model_bits,
            pass_linear_list=kwargs.pop('pass_linear_list', []),
            use_prune=kwargs.pop('use_prune', False),
            sec_obj=self.sec_obj,
            sec_obj_range=self.sec_obj_range,
            config=self.config,
        )
        self.ga_pop_size = kwargs.pop('ga_pop_size', 40)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        self.debug = kwargs.pop('debug', False)
        self.ga_algorithm = kwargs.pop('ga_algorithm', 'nsga2')
        self.nan_value = kwargs.pop('nan_value', 50)
        
        self.sec_iterations = kwargs.pop('sec_iterations', 0)  # number of iterations to run search
        self.sec_n_doe = kwargs.pop('sec_n_doe', 50)  # number of architectures to train before fit surrogate model
        self.sec_n_iter = kwargs.pop('sec_n_iter', 20)  # number of architectures to train in each iteration
        self.sec_metric = kwargs.pop('sec_metric', 'ppl')  # number of architectures to train in each iteration
        self.sec_ga_pop_size = kwargs.pop('sec_ga_pop_size', 20)

    def search(self):
        total_start = time()
        start_it = 1
        
        if self.resume:
            archive, start_it = self._resume_from_dir()

        else:
            # the following lines corresponding to Algo 1 line 1-7 in the paper
            archive = []  # initialize an empty archive to store all trained CNNs

            # Design Of Experiment
            if self.iterations < 1:
                arch_doe = self.search_space.sample(
                    n_samples=self.n_doe,
                    q=[self.small_model_bits, self.large_model_bits],
                    k=[self.small_model_bits, self.large_model_bits],
                    v=[self.small_model_bits, self.large_model_bits],
                    o=[self.small_model_bits, self.large_model_bits],
                    down=[self.small_model_bits, self.large_model_bits],
                    up=[self.small_model_bits, self.large_model_bits],
                    gate=[self.small_model_bits, self.large_model_bits],
                    pool=[x[0] for x in archive])
            else:
                arch_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])

            # parallel evaluation of arch_doe
            metric, complexity = self._evaluate(arch_doe)

            # store evaluated / trained architectures
            for member in zip(arch_doe, metric, complexity):
                archive.append(member)

        # reference point (nadir point) for calculating hypervolume
        ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])
        print(f'data preparation time : {time() - total_start:.2f}s')

        # main loop of the search
        for it in range(start_it, self.iterations + self.sec_iterations + 1):
            print(self.args)
            n_candidates = self.n_iter if it <= self.iterations else self.sec_n_iter
            self.cur_ga_pop_size = self.ga_pop_size if it <= self.iterations else self.sec_ga_pop_size

            iter_start = time()
            if it == self.iterations + 1:
                print(f'start second search')
                if self.ga_algorithm == 'nsga2':
                    F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
                    front = NonDominatedSorting().do(F, only_non_dominated_front=True)

                elif self.ga_algorithm == 'ga':
                    front = np.argsort([x[1] for x in archive])[:self.sec_n_doe]
                
                # arch_doe = [x[0] for x in archive][front]
                arch_doe = [x[0] for x in archive]
                arch_doe = [arch_doe[i] for i in front]
                
                archive.clear()
                self.metric = self.sec_metric

                metric, complexity = self._evaluate(arch_doe)

                # store evaluated / trained architectures
                for member in zip(arch_doe, metric, complexity):
                    archive.append(member)

            # construct accuracy predictor surrogate model from archive
            # Algo 1 line 9 / Fig. 3(a) in the paper
            metric_predictor, a_metric_pred = self._fit_acc_predictor(archive, device=self.evaluator.device)

            # search for the next set of candidates for high-fidelity evaluation (lower level)
            # Algo 1 line 10-11 / Fig. 3(b)-(d) in the paper
            candidates, c_metric_pred = self._next(archive, metric_predictor, n_candidates)

            # high-fidelity evaluation (lower level)
            # Algo 1 line 13-14 / Fig. 3(e) in the paper
            c_metric, complexity = self._evaluate(candidates)

            # check for accuracy predictor's performance
            rmse, rho, tau = get_correlation(
                np.vstack((a_metric_pred, c_metric_pred)), np.array([x[1] for x in archive] + c_metric))

            # add to archive
            # Algo 1 line 15 / Fig. 3(e) in the paper
            for member in zip(candidates, c_metric, complexity):
                # if self.sec_obj_range[0] <= member[2] and member[2] <= self.sec_obj_range[1]:
                archive.append(member)

            # calculate hypervolume
            hv = self._calc_hv(
                ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))

            iter_time = time() - iter_start
            # print iteration-wise statistics
            print(f"Iter {it}: hv = {hv:.2f}, iter time : {(time() - iter_start):.2f}s")
            print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall’s Tau = {tau:.4f}")
            print(f'iteration time : {iter_time:.2f}s')

            # dump the statistics
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                json.dump({'archive': archive, 'candidates': archive[-n_candidates:], 'hv': hv,
                           'surrogate': {
                               'model': self.predictor, 'name': metric_predictor.name,
                               'winner': metric_predictor.winner if self.predictor == 'as' else metric_predictor.name,
                               'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time}, 'iteration' : it}, handle)
            if self.debug:
                from pymoo.visualization.scatter import Scatter
                # plot
                plot = Scatter(legend={'loc': 'lower right'})
                F = np.full((len(archive), 2), np.nan)
                F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                # F[:, 1] = 100 - np.array([x[1] for x in archive])  # performance
                F[:, 1] = np.array([x[1] for x in archive])  # performance
                plot.add(F, s=5, facecolors='none', edgecolors='b', label='archive')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                # F[:, 1] = 100 - np.array(c_metric)
                F[:, 1] = np.array(c_metric)
                plot.add(F, s=10, color='r', label='candidates evaluated')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = c_metric_pred[:, 0]
                plot.add(F, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
                plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
        total_time_elapsed = time() - total_start
        print(f'total time elapsed : {total_time_elapsed:.2f}s')

        sentences = []
        for k, v in self.args.items():
            sentences.append(f"{k}: {v}\n")
        sentences.append(f'Total time: {total_time_elapsed:.2f}s')
        # sentences.append("\n")

        with open(os.path.join(self.save_path, self.result_file), 'w') as f:
            for sentence in sentences:
                f.write(sentence)

        print(self.args)
        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """

        with open(self.resume, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive']
            it = resume_file['archive']

        return archive, it + 1

    def _evaluate(self, archs):
        metric_list, complexity_list = [], []
        for arch in tqdm(archs, desc='Eval Arch'):
            metric, complexity = self.evaluator.eval(arch, self.metric)
            metric_list.append(np.nan_to_num(metric[self.dataset], nan=self.nan_value))
            complexity_list.append(complexity[self.sec_obj])

        return metric_list, complexity_list
        # gen_dir = os.path.join(self.save_path, "iter_{}".format(it))
        # prepare_eval_folder(
        #     gen_dir, archs, self.gpu, self.n_gpus, data=self.data, dataset=self.dataset,
        #     n_classes=self.n_classes, supernet_path=self.supernet_path,
        #     num_workers=self.n_workers, valid_size=self.vld_size,
        #     trn_batch_size=self.trn_batch_size, vld_batch_size=self.vld_batch_size,
        #     n_epochs=self.n_epochs, test=self.test, latency=self.latency, verbose=False)

        # subprocess.call("sh {}/run_bash.sh".format(gen_dir), shell=True)

        # for i in range(len(archs)):
            # try:
            #     stats = json.load(open(os.path.join(gen_dir, "net_{}.stats".format(i))))
            # except FileNotFoundError:
            #     # just in case the subprocess evaluation failed
            #     stats = {'top1': 0, self.sec_obj: 1000}  # makes the solution artificially bad so it won't survive
            #     # store this architecture to a separate in case we want to revisit after the search
            #     os.makedirs(os.path.join(self.save_path, "failed"), exist_ok=True)
            #     shutil.copy(os.path.join(gen_dir, "net_{}.subnet".format(i)),
            #                 os.path.join(self.save_path, "failed", "it_{}_net_{}".format(it, i)))

            # ppl.append(100 - stats['top1'])
            # complexity.append(stats[self.sec_obj])


    def _fit_acc_predictor(self, archive, device='cpu'):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        # assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        metric_predictor = get_acc_predictor(self.predictor, inputs, targets, device=device)

        return metric_predictor, metric_predictor.predict(inputs, device=device)

    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        # F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        # front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # # non-dominated arch bit-strings
        # nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initialize the candidate finding optimization problem

        # initiate a multi-objective solver to optimize the problem
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        if self.ga_algorithm == 'nsga2':
            # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
            # get non-dominated architectures from archive
            front = NonDominatedSorting().do(F, only_non_dominated_front=True)
            # non-dominated arch bit-strings
            nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

            method = NSGA2(pop_size=self.cur_ga_pop_size, sampling=nd_X,  # initialize with current nd archs
                # crossover=TwoPointCrossover(prob=0.9),
                crossover=BinomialCrossover(prob=0.9, n_offsprings=1),
                # crossover=BinomialCrossover(prob=0.9, n_offsprings=2),
                # crossover=MyTwoPointCrossover(prob=0.9, n_offsprings=1),
                # mutation=IntPolynomialMutation(eta=1.0),
                mutation=IntPolynomialMutation(prob=0.1, eta=1.0),
                # mutation=IntPolynomialMutation(prob=0.2, eta=1.0),
                # mutation=IntPolynomialMutation(prob=0.02, eta=1.0),
                eliminate_duplicates=True)
            
            problem = AuxiliarySingleLevelProblem(self.search_space, predictor, self.config, self.sec_obj_range, self.sec_obj)
            
        elif self.ga_algorithm == 'ga':
            # Selecting top k metrics in archive
            front = np.argsort([x[1] for x in archive])[:self.cur_ga_pop_size]
            nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

            method = GA(pop_size=self.cur_ga_pop_size, sampling=nd_X,  # initialize with current nd archs
                # crossover=TwoPointCrossover(prob=0.9),
                crossover=BinomialCrossover(prob=0.9, n_offsprings=1),
                # crossover=MyTwoPointCrossover(prob=0.9, n_offsprings=1),
                # crossover=MyUniformCrossover(prob=0.9, n_offsprings=1),
                mutation=IntPolynomialMutation(eta=1.0),
                eliminate_duplicates=True)
            
            problem=ConstrainedSelectionProblem(self.search_space, predictor, self.config, self.sec_obj_range, self.sec_obj)
        else:
            raise NotImplementedError(f'{self.ga_algorithm} is not implemented')

        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1], K, self.subset_pop_size)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(pop.get("X"))

    # @staticmethod
    def _subset_selection(self, pop, nd_F, K, pop_size):
        candidates = np.array([get_net_info(self.search_space.decode(x), self.config)[self.sec_obj] for x in pop.get("X")])
        problem = SubsetProblem(candidates, nd_F, K)
        # problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
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

    def __init__(self, search_space, predictor, config, sec_obj_range, sec_obj='bits', xl=1, xu=2):
        super().__init__(n_var=int(config['n_block']) * int(config['n_linear']), n_obj=2, n_constr=2, type_var=int)

        self.ss = search_space
        self.predictor = predictor
        self.xl = xl * np.ones(self.n_var)
        self.xu = xu * np.ones(self.n_var)
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        # self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}
        self.config = config

        for pass_linear in self.ss.pass_linear_list:
            blk, module, linear = pass_linear.split('.')
            linear_idx = config['linear'].index(f'{module}.{linear}')
            self.xl[int(blk) * config['n_linear'] + linear_idx] = len(getattr(self.ss, f'{linear}_option')) - 1

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        metrics = self.predictor.predict(x)[:, 0]

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            arch = self.ss.decode(_x)
            info = get_net_info(arch, self.config)
            f[i, 0] = metric
            f[i, 1] = info[self.sec_obj]
            g[i, 0] = 1 - info[self.sec_obj] / self.sec_obj_range[0]
            g[i, 1] = info[self.sec_obj] / self.sec_obj_range[1] - 1

        out["F"] = f
        out["G"] = g

class ConstrainedSelectionProblem(Problem):
    """  """

    def __init__(self, search_space, predictor, config, sec_obj_range, sec_obj='bits', xl=1, xu=2):
        super().__init__(n_var=int(config['n_block']) * int(config['n_linear']), n_obj=1, n_constr=2, type_var=int)
        self.ss = search_space
        self.predictor = predictor
        # self.xl = np.zeros(self.n_var)
        self.xl = xl * np.ones(self.n_var)
        self.xu = xu * np.ones(self.n_var)
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        # self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}
        self.config = config

        for pass_linear in self.ss.pass_linear_list:
            blk, module, linear = pass_linear.split('.')
            linear_idx = config['linear'].index(f'{module}.{linear}')
            self.xl[int(blk) * config['n_linear'] + linear_idx] = len(getattr(self.ss, f'{linear}_option')) - 1

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        metrics = self.predictor.predict(x)[:, 0]

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            arch = self.ss.decode(_x)
            info = get_net_info(arch, self.config)
            f[i, 0] = metric
            g[i, 0] = 1 - info[self.sec_obj] / self.sec_obj_range[0]
            g[i, 1] = info[self.sec_obj] / self.sec_obj_range[1] - 1

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
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g


def main(args):
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = Search(vars(args))
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=50,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    # parser.add_argument('--n_gpus', type=int, default=8,
    #                     help='total number of available gpus')
    # parser.add_argument('--gpu', type=int, default=1,
    #                     help='number of gpus per evaluation job')
    # parser.add_argument('--n_workers', type=int, default=4,
    #                     help='number of workers for dataloader per evaluation job')
    # parser.add_argument('--vld_size', type=int, default=None,
    #                     help='validation set size, randomly sampled from training set')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_bits', type=float, default=4,
                        help='test batch size for inference')
    parser.add_argument('--small_model_path', type=str, default='',
                        help='file path to the small quantized model')
    parser.add_argument('--small_model_bits', type=float, default=2,
                        help='bits of the small quantized model')
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
    # parser.add_argument('--predictor_data', type=str, default='',
    #                     help='dataset to pretrain the predictor')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='config file to read the model meta data')
    parser.add_argument('--ga_pop_size', type=int, default=40,
                        help='population size of the NSGA stage')
    parser.add_argument('--subset_pop_size', type=int, default=100,
                        help='population size of the subset selection stage')
    parser.add_argument('--debug', type=bool, default=False,
                        help='visualization of each iteration results')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--ga_algorithm', type=str, default='nsga2',
                        help='')
    parser.add_argument('--quant_method', type=str, default='',
                        help='')
    parser.add_argument('--nan_value', type=float, default=50,
                        help='')
    parser.add_argument('--sec_iterations', type=int, default=0,
                        help='number of search iterations')
    parser.add_argument('--sec_n_doe', type=int, default=224,
                        help='initial sample size for DOE')
    parser.add_argument('--sec_n_iter', type=int, default=20,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--sec_metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss)')
    parser.add_argument('--sec_ga_pop_size', type=int, default=50,
                        help='population size of the NSGA stage')
    
    cfgs = parser.parse_args()
    main(cfgs)

