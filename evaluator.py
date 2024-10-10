import os
import json
import torch
import argparse
import numpy as np

import math
import gc
from copy import deepcopy

from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from gptqmodel import GPTQModel
from gptqmodel.utils import get_backend
from utils.owq.utils.modelutils import load_model
from utils.func_utils import setsubattr, getsubattr, getblock, get_net_info
from utils.data_utils import get_loader
from utils.eval_utils import eval_metric

import warnings
warnings.simplefilter("ignore")

class LlamaEvaluator:
    def __init__(self,
                 config,
                 quant_method='',
                 model_name='',
                 large_model_path='',
                 large_model_bits=4,
                 small_model_path='',
                 small_model_bits=2,
                 datasets=['wikitext2'],
                 seed=0,
                 seqlen=2048,
                 n_sample=128,
                 device_map='auto',
                 cache_dir=None):

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)
        # device = torch.device("cuda:0")
        self.device = next(self.model.parameters()).device
        if quant_method == 'hqq':
            self.large_model = AutoHQQHFModel.from_quantized(large_model_path, device_map=device_map) # .to(self.device)
            self.small_model = AutoHQQHFModel.from_quantized(small_model_path, device_map=device_map) # .to(self.device)
        elif quant_method == 'gptq':
            self.large_model = GPTQModel.from_quantized(large_model_path, device_map=device_map, backend=get_backend('AUTO')).model
            self.small_model = GPTQModel.from_quantized(small_model_path, device_map=device_map, backend=get_backend('AUTO')).model
        elif quant_method == 'owq' :
            self.large_model = load_model(model_name, large_model_path, device=device_map)
            self.small_model = load_model(model_name, small_model_path, device=device_map)
        else:
            raise NotImplementedError(f"{quant_method} is not supported")

        self.large_model_bits = large_model_bits
        self.small_model_bits = small_model_bits
        self.config = config
        self.seqlen = seqlen
        self.train_loaders = {dataset: get_loader(dataset, model=model_name, n_sample=n_sample, train=True, seed=seed, seqlen=seqlen) for dataset in datasets}
        self.test_loaders = {dataset: get_loader(dataset, model=model_name, train=False, seqlen=seqlen) for dataset in datasets}

        self.model.eval()
        self.large_model.eval()
        self.small_model.eval()
        
        self.model.use_cache = False
        self.large_model.use_cache = False
        self.small_model.use_cache = False

    def sample(self, arch):
        self.validate_arch(arch)
        for linear, linear_bits in arch.items():
            for blk_idx, bits in enumerate(linear_bits):
                if bits == 0:
                    raise NotImplementedError(f'0 bit are not supported currently.')
                elif math.isclose(bits, self.small_model_bits):
                    setsubattr(getblock(self.model, self.config, blk_idx), linear, deepcopy(getsubattr(getblock(self.small_model, self.config, blk_idx), linear)))
                elif math.isclose(bits, self.large_model_bits):
                    setsubattr(getblock(self.model, self.config, blk_idx), linear, deepcopy(getsubattr(getblock(self.large_model, self.config, blk_idx), linear)))
                else:
                    raise NotImplementedError(f'Only 0, {self.small_model_bits}, {self.large_model_bits} bits are allowed, current linear : {linear}, {linear_bits}')
                
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

    def eval(self, arch, metric='ppl'):
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        else:
            NotImplementedError(f"metric should be 'ppl' or 'loss', not {metric}")
        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=self.sample(arch), metric=metric, loader=loader, device=self.device, seqlen=self.seqlen)
        complexity = get_net_info(arch, self.config)
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


def main(args):
    """ one evaluation of a subnet or a config from a file """
    mode = 'subnet'
    if args.config is not None:
        if args.init is not None:
            mode = 'config'

    print('Evaluation mode: {}'.format(mode))
    if mode == 'config':
        net_config = json.load(open(args.config))
        subnet = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_connect_rate)
        init = torch.load(args.init, map_location='cpu')['state_dict']
        subnet.load_state_dict(init)
        subnet.classifier.dropout_rate = args.drop_rate
        try:
            resolution = net_config['resolution']
        except KeyError:
            resolution = args.resolution

    elif mode == 'subnet':
        config = json.load(open(args.subnet))
        evaluator = OFAEvaluator(n_classes=args.n_classes, model_path=args.supernet_path)
        subnet, _ = evaluator.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
        resolution = config['r']

    else:
        raise NotImplementedError

    OFAEvaluator.eval(
        subnet, log_dir=args.log_dir, data_path=args.data, dataset=args.dataset, n_epochs=args.n_epochs,
        resolution=resolution, trn_batch_size=args.trn_batch_size, vld_batch_size=args.vld_batch_size,
        num_workers=args.num_workers, valid_size=args.valid_size, is_test=args.test, measure_latency=args.latency,
        no_logs=(not args.verbose), reset_running_statistics=args.reset_running_statistics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--large_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_bits', type=int, default=4,
                        help='test batch size for inference')
    parser.add_argument('--small_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--small_model_bits', type=int, default=2,
                        help='test batch size for inference')
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    
    parser.add_argument('--log_dir', type=str, default='.tmp',
                        help='directory for logging')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    # parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
    #                     help='file path to supernet weights')
    # parser.add_argument('--subnet', type=str, default=None,
    #                     help='location of a json file of ks, e, d, and e')
    parser.add_argument('--config', type=str, default=None,
                        help='location of a json file of specific model declaration')
    # parser.add_argument('--num_workers', type=int, default=6,
    #                     help='number of workers for data loading')
    parser.add_argument('--save', type=str, default=None,
                        help='location to save the evaluated metrics')
    parser.add_argument('--latency', type=str, default=None,
                        help='latency measurement settings (gpu64#cpu)')
    parser.add_argument('--arch', type=str, default="",
                        help='')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to display evaluation progress')
    # parser.add_argument('--save_config', action='store_true', default=False,
    #                     help='save config file')
    cfgs = parser.parse_args()

    # cfgs.teacher_model = None

    main(cfgs)


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

