import os
import json
import torch
import argparse
import numpy as np

import math
import gc
from copy import deepcopy
from tqdm import tqdm
from time import time

from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import prepare_for_inference

# from gptqmodel import GPTQModel
# from gptqmodel.utils import get_backend

from utils.owq.utils.modelutils import load_model
from utils.func_utils import setsubattr, getsubattr, getblock, get_net_info
from utils.data_utils import get_loader
from utils.eval_utils import eval_metric, get_logits
# from awq.quantize.quantizer import real_quantize_model_weight

from model.skip_llama import block_replace

import warnings
warnings.simplefilter("ignore")

class LlamaEvaluator:
    def __init__(self,
                 config,
                 method='',
                 model_name='',
                 quant_model_paths=[],
                 quant_model_bits=[],
                 datasets=['wikitext2'],
                 seed=0,
                 seqlen=2048,
                 n_sample=128,
                 device_map='auto',
                 cache_dir=None,
                 loss_func='cross_entropy'):
        
        self.method = method
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)
        # device = torch.device("cuda:0")
        if method == 'hqq':
            # self.model = AutoHQQHFModel.from_quantized(large_model_path, device_map=device_map) # .to(self.device)
            # self.large_model = AutoHQQHFModel.from_quantized(large_model_path, device_map=device_map) # .to(self.device)
            # self.small_model = AutoHQQHFModel.from_quantized(small_model_path, device_map=device_map) # .to(self.device)
            self.quant_models = [AutoHQQHFModel.from_quantized(path, device_map=device_map) for path in quant_model_paths]
            self.quant_model_bits = quant_model_bits
        elif method == 'gptq':
            # self.model = GPTQModel.from_quantized(large_model_path, device_map=device_map, backend=get_backend('AUTO')).model
            # self.large_model = GPTQModel.from_quantized(large_model_path, device_map=device_map, backend=get_backend('AUTO')).model
            # self.small_model = GPTQModel.from_quantized(small_model_path, device_map=device_map, backend=get_backend('AUTO')).model
            self.quant_models = [GPTQModel.from_quantized(path, device_map=device_map, backend=get_backend('AUTO')).model for path in quant_model_paths]
            self.quant_model_bits = quant_model_bits
        elif method == 'owq' :
            # self.model = load_model(model_name, large_model_path, device=device_map)
            # self.large_model = load_model(model_name, large_model_path, device=device_map)
            # self.small_model = load_model(model_name, small_model_path, device=device_map)
            self.quant_models = [load_model(model_name, path, device=device_map) for path in quant_model_paths]
            self.quant_model_bits = quant_model_bits
        elif method == 'awq' :
            # self.model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map)
            # self.large_model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map)
            # self.small_model = AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map)
            # self.model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map)
            # self.large_model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map)
            # self.small_model = AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map)
            self.quant_models = [AutoModelForCausalLM.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map) for path in quant_model_paths]
            self.quant_model_bits = quant_model_bits
        elif method == 'layer_prune':
            self.model = block_replace(self.model)
        else:
            raise NotImplementedError(f"{method} is not supported")
        self.device = next(self.model.parameters()).device

        self.config = config
        self.seqlen = seqlen
        self.train_loaders = {dataset: get_loader(dataset, model=model_name, n_sample=n_sample, train=True, seed=seed, seqlen=seqlen) for dataset in datasets}
        self.test_loaders = {dataset: get_loader(dataset, model=model_name, train=False, seqlen=seqlen) for dataset in datasets}

        self.model.eval()
        self.model.use_cache = False
        if method != 'layer_prune':
            for q_model in self.quant_models:
                q_model.eval()
                q_model.use_cache = False
        self.loss_func = loss_func
        self.dense_logits = {dataset: (get_logits(self.model, loader, bs=1, seqlen=seqlen, device=self.device) if self.loss_func == 'jsd' else None) for dataset, loader in self.train_loaders.items()}

    def sample(self, arch):
        # self.validate_arch(arch)
        if self.method in ['hqq', 'awq', 'owq', 'gptq'] :
            for linear, linear_bits in arch.items():
                for blk_idx, bits in enumerate(linear_bits):
                    flag = False
                    for q_bits, q_model in zip(self.quant_model_bits, self.quant_models):
                        if math.isclose(bits, q_bits):
                            setsubattr(getblock(self.model, self.config, blk_idx), linear, deepcopy(getsubattr(getblock(q_model, self.config, blk_idx), linear)))
                            flag = True
                    if not flag:
                        raise NotImplementedError(f'{linear}: {linear_bits} is not available')
                    # if bits == 0:
                    #     raise NotImplementedError(f'0 bit are not supported currently.')
                    # elif math.isclose(bits, self.small_model_bits):
                    #     setsubattr(getblock(self.model, self.config, blk_idx), linear, deepcopy(getsubattr(getblock(self.small_model, self.config, blk_idx), linear)))
                    # elif math.isclose(bits, self.large_model_bits):
                    #     setsubattr(getblock(self.model, self.config, blk_idx), linear, deepcopy(getsubattr(getblock(self.large_model, self.config, blk_idx), linear)))
                    # else:
                    #     raise NotImplementedError(f'Only 0, {self.small_model_bits}, {self.large_model_bits} bits are allowed, current linear : {linear}, {linear_bits}')
        elif self.method == 'layer_prune':
            for layer, layer_arch in arch.items():
                for blk_idx, a in enumerate(layer_arch):
                    if a == 0:
                        if layer == 'self_attn':
                            getblock(self.model, self.config, blk_idx).skip_attn(reuse=True)
                        elif layer == 'mlp':
                            getblock(self.model, self.config, blk_idx).skip_mlp(reuse=True)
                    elif a == 1:
                        if layer == 'self_attn':
                            getblock(self.model, self.config, blk_idx).use_attn()
                        elif layer == 'mlp':
                            getblock(self.model, self.config, blk_idx).use_mlp()
        
        gc.collect()
        return self.model
    
    def validate_arch(self, arch):
        assert all([l in self.config['linear'] for l in list(arch.keys())]), f'{list(arch.keys())} are invalid'
        for linear, linear_bits in arch.items():
            assert len(linear_bits) == self.config['n_block'], f'{linear}: len(linear_bits) != n_block'
            _, linear = linear.split('.')
            assert all([b in [0, self.small_model_bits, self.large_model_bits] for b in linear_bits]), f'{linear}: {linear_bits} are not compatible with the evaluator.'

    # @staticmethod
    # def save_net_config(path, net, config_name='net.config'):
    #     """ dump run_config and net_config to the model_folder """
    #     net_save_path = os.path.join(path, config_name)
    #     json.dump(net.config, open(net_save_path, 'w'), indent=4)
    #     print('Network configs dump to %s' % net_save_path)

    # @staticmethod
    # def save_net(path, net, model_name):
    #     """ dump net weight as checkpoint """
    #     if isinstance(net, torch.nn.DataParallel):
    #         checkpoint = {'state_dict': net.module.state_dict()}
    #     else:
    #         checkpoint = {'state_dict': net.state_dict()}
    #     model_path = os.path.join(path, model_name)
    #     torch.save(checkpoint, model_path)
    #     print('Network model dump to %s' % model_path)

    def eval(self, arch, metric, method, loss_func='cross_entropy'):
        # if metric == 'latency':
        #     measure_latency(model=self.sample(arch))
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        else:
            NotImplementedError(f"metric should be 'ppl' or 'loss', not {metric}")
        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=self.sample(arch), metric=metric, loader=loader, device=self.device, seqlen=self.seqlen, loss_func=loss_func, dense_logits_list=self.dense_logits[dataset])
        complexity = get_net_info(arch, self.config, method)
        torch.cuda.empty_cache()
        return metric_list, complexity
        # lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        # info = get_net_info(
        #     subnet, (3, resolution, resolution), measure_latency=measure_latency,
        #     print_info=False, clean=True, lut=lut)

        # run_config = get_run_config(
        #     dataset=dataset, data_path=data_path, image_size=resolution, n_epochs=n_epochs,
        #     train_batch_size=trn_batch_size, test_batch_size=vld_batch_size,
        #     n_worker=num_workers, valid_size=valid_size)

        # # set the image size. You can set any image size from 192 to 256 here
        # run_config.data_provider.assign_active_img_size(resolution)

        # if n_epochs > 0:
        #     # for datasets other than the one supernet was trained on (ImageNet)
        #     # a few epochs of training need to be applied
        #     subnet.reset_classifier(
        #         last_channel=subnet.classifier.in_features,
        #         n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)

        # run_manager = RunManager(log_dir, subnet, run_config, init=False)
        # if reset_running_statistics:
        #     # run_manager.reset_running_statistics(net=subnet, batch_size=vld_batch_size)
        #     run_manager.reset_running_statistics(net=subnet)

        # if n_epochs > 0:
        #     subnet = run_manager.train(cfgs)

        # loss, top1, top5 = run_manager.validate(net=subnet, is_test=is_test, no_logs=no_logs)

        # info['loss'], info['top1'], info['top5'] = loss, top1, top5

        # save_path = os.path.join(log_dir, 'net.stats') if cfgs.save is None else cfgs.save
        # if cfgs.save_config:
        #     OFAEvaluator.save_net_config(log_dir, subnet, "net.config")
        #     OFAEvaluator.save_net(log_dir, subnet, "net.init")
        # with open(save_path, 'w') as handle:
        #     json.dump(info, handle)

        # print(info)


# def measure_latency(args):
#     print(args)

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

#     with open(args.config, 'r') as f:
#         config = json.load(f)[args.model_name]

#     evaluator = LlamaEvaluator(
#         config=config,
#         model_name=args.model_name,
#         method=args.method,
#         large_model_path=args.large_model_path,
#         large_model_bits=args.large_model_bits,
#         small_model_path=args.small_model_path,
#         small_model_bits=args.small_model_bits,
#         seqlen=args.seqlen,
#         n_sample=args.n_sample,
#         datasets=[args.dataset]
#     )
#     ppl_archive = list()

#     # complexity_list = list()
#     # archs = search_space.initialize(args.n_data)
#     with open(args.arch_file, 'r') as f:
#         archive = json.load(f)['archive']
#         archs = [x[0] for x in archive]

#     for arch in tqdm(archs):
#         iter_start = time()

#         model = deepcopy(evaluator.sample(arch))
#         if args.method == 'hqq':
#             prepare_for_inference(model, backend='bitblas')

        # test_measure_latency = 
        
#         # ppl = eval_metric(model=model, metric='ppl', loader=evaluator.test_loaders, device=evaluator.device, seqlen=evaluator.seqlen)
#         complexity = get_net_info(arch, config)
#         ppl_archive.append([arch, ppl[args.dataset], complexity[args.sec_obj]])

#         iter_time = time() - iter_start

#         print(f'{arch} {args.sec_obj}: {complexity[args.sec_obj]:.3f}, ppl : {ppl:2f}, time : {iter_time:.2f}s')
#         # print(f'{args.sec_obj}: {complexity[args.sec_obj]:.3f}, ppl : {ppl[args.dataset]:.2f}, loss : {loss[args.dataset]:.2f}, time : {iter_time:.2f}s')
#         # complexity_list.append(complexity)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, default='',
#                         help='file path to supernet weights')
#     parser.add_argument('--method', type=str, default='',
#                         help='')
#     parser.add_argument('--large_model_path', type=str, default='',
#                         help='file path to supernet weights')
#     parser.add_argument('--large_model_bits', type=float, default=4,
#                         help='test batch size for inference')
#     parser.add_argument('--small_model_path', type=str, default='',
#                         help='file path to supernet weights')
#     parser.add_argument('--small_model_bits', type=float, default=2,
#                         help='test batch size for inference')
#     parser.add_argument('--dataset', type=str, default='wikitext2',
#                         help='dataset')
#     parser.add_argument('--seed', type=int, default=0,
#                         help='test batch size for inference')
#     parser.add_argument('--n_sample', type=int, default=128,
#                         help='test batch size for inference')
#     parser.add_argument('--seqlen', type=int, default=2048,
#                         help='test batch size for inference')
#     parser.add_argument('--metric', type=str, default='ppl',
#                         help='which accuracy predictor model to fit (ppl/loss)')
#     parser.add_argument('--pass_linear_list', type=str, nargs='+', default=[], 
#                         help='which accuracy predictor model to fit (ppl/loss)')
#     parser.add_argument('--config', type=str, default='config/llama.json',
#                         help='')
#     parser.add_argument('--n_data', type=int, default=1000,
#                         help='test batch size for inference')
#     parser.add_argument('--loss_json_file', type=str, default='',
#                         help='')
#     parser.add_argument('--ppl_json_file', type=str, default='',
#                         help='')
#     parser.add_argument('--sec_obj', type=str, default='bits',
#                         help='second objective to optimize simultaneously')
#     parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
#                         help='')
#     parser.add_argument('--nan_value', type=float, default=50,
#                         help='')
#     parser.add_argument('--metric', type=float, nargs='+', default=[], 
#                         help='\'latency\', \'ppl\'')
    
#     cfgs = parser.parse_args()
#     eval_arch(cfgs)


# def parse_string_list(string):
#     if isinstance(string, str):
#         # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
#         return list(map(int, string[1:-1].split()))
#     else:
#         return string


# def pad_none(x, depth, max_depth):
#     new_x, counter = [], 0
#     for d in depth:
#         for _ in range(d):
#             new_x.append(x[counter])
#             counter += 1
#         if d < max_depth:
#             new_x += [None] * (max_depth - d)
#     return new_x


# def get_net_info(net, data_shape, measure_latency=None, print_info=True, clean=False, lut=None):

#     net_info = utils.get_net_info(
#         net, data_shape, measure_latency, print_info=print_info, clean=clean, lut=lut)

#     gpu_latency, cpu_latency = None, None
#     for k in net_info.keys():
#         if 'gpu' in k:
#             gpu_latency = np.round(net_info[k]['val'], 2)
#         if 'cpu' in k:
#             cpu_latency = np.round(net_info[k]['val'], 2)

#     return {
#         'params': np.round(net_info['params'] / 1e6, 2),
#         'flops': np.round(net_info['flops'] / 1e6, 2),
#         'gpu': gpu_latency, 'cpu': cpu_latency
#     }


# def validate_config(config, max_depth=4):
#     kernel_size, exp_ratio, depth = config['ks'], config['e'], config['d']

#     if isinstance(kernel_size, str): kernel_size = parse_string_list(kernel_size)
#     if isinstance(exp_ratio, str): exp_ratio = parse_string_list(exp_ratio)
#     if isinstance(depth, str): depth = parse_string_list(depth)

#     assert (isinstance(kernel_size, list) or isinstance(kernel_size, int))
#     assert (isinstance(exp_ratio, list) or isinstance(exp_ratio, int))
#     assert isinstance(depth, list)

#     if len(kernel_size) < len(depth) * max_depth:
#         kernel_size = pad_none(kernel_size, depth, max_depth)
#     if len(exp_ratio) < len(depth) * max_depth:
#         exp_ratio = pad_none(exp_ratio, depth, max_depth)

#     # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
#     return {'ks': kernel_size, 'e': exp_ratio, 'd': depth}

