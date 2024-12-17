import torch
import numpy as np

import math
import gc
from copy import deepcopy
from tqdm import tqdm
from time import time

from transformers import AutoModelForCausalLM
from utils.func import *
from utils.data import get_loader
from utils.eval import eval_metric, get_logits
from utils.dispatch import simple_dispatch_model

from model.skip_llama import block_replace

import warnings
warnings.simplefilter("ignore")

class LlamaEvaluator:
    def __init__(self,  
                 config,
                 accelerator,
                 method=[],
                 model_id='',
                 quant_model_paths=[],
                 quant_model_bits=[],
                 outlier=None,
                 datasets=['wikitext2'],
                 seed=0,
                 seqlen=2048,
                 n_sample=128,
                 device_map='auto',
                 cache_dir=None,
                 loss_func='cross_entropy',
                 latency_table=None):
        
        # model_id = os.path.join(model_path, model_name)
        self.method = method
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)

        with accelerator.main_process_first():
            self.train_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, train=True, seed=seed, seqlen=seqlen)) for dataset in datasets}
            self.test_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, train=False, seqlen=seqlen)) for dataset in datasets}

        self.loss_func = loss_func
        self.outlier = dict()
        if loss_func == 'jsd' or outlier is not None:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map=device_map, low_cpu_mem_usage=True)

            if loss_func == 'jsd':
                self.dense_logits = {dataset: get_logits(model, loader) for dataset, loader in self.train_loaders.items()}

            if outlier is not None:
                for blk_idx in range(int(config['n_block'])):
                    for linear_group in config['linear']:
                        for linear in linear_group.split(','):
                            key = f'{config["layers"]}.{blk_idx}.{linear}'
                            if key in outlier:
                                self.outlier[f'{blk_idx}.{linear}'] = [outlier[key], get_fp16_channel(getsubattr(getblock(model, config)[blk_idx], linear), outlier[key])]
                            
            del model; gc.collect(); torch.cuda.empty_cache()

        if loss_func != 'jsd':
            self.dense_logits = {dataset: None for dataset in self.train_loaders.keys()}

        self.quant_models = list()
        if 'hqq' in method:
            with accelerator.main_process_first():
                self.model = load_hqq_model(quant_model_paths[np.argmax(quant_model_bits)], device_map)
                import pdb; pdb.set_trace()
                self.remove_linears(self.model, config)
                self.quant_models = [load_hqq_model(p, device_map) for p in quant_model_paths]
            self.quant_model_bits = quant_model_bits

        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map, cache_dir=cache_dir)

        if 'layer_prune' in method:
            self.model = block_replace(self.model)
            self.model = simple_dispatch_model(self.model, device_map)

        self.config = config
        self.latency_table = latency_table
        self.seqlen = seqlen
            
        self.model.eval()
        self.model.use_cache = False
        for q_model in self.quant_models:
            q_model.eval()
            q_model.use_cache = False

        accelerator.wait_for_everyone()

    def sample(self, arch, reuse=True):
        # from time import time
        # sample_start = time()
        # self.validate_arch(arch)
        if 'hqq' in self.method or 'awq' in self.method or 'gptq' in self.method or 'owq' in self.method:
            for linear_group, linear_group_bits in arch['linear'].items():
                for blk_idx, bits in enumerate(linear_group_bits):
                    flag = False
                    for q_bits, q_model in zip(self.quant_model_bits, self.quant_models):
                        # if math.isclose(bits, q_bits):
                        if math.isclose(int(bits), q_bits):
                            for linear in linear_group.split(','):
                                # setsubattr(getblock(self.model, self.config)[blk_idx], linear, deepcopy(getsubattr(getblock(q_model, self.config)[blk_idx], linear)))
                                setsubattr(getblock(self.model, self.config)[blk_idx], linear, getsubattr(getblock(q_model, self.config)[blk_idx], linear))
                            flag = True

                    if not math.isclose(bits - int(bits), 0):
                        for linear in linear_group.split(','):
                            # insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{self.config["layers"]}.{blk_idx}.{linear}'])
                            insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{blk_idx}.{linear}'])
                    else:
                        for linear in linear_group.split(','):
                            remove_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear))

                    if not flag:
                        raise NotImplementedError(f'{linear_group}: {linear_group_bits} is not available')

        if 'layer_prune' in self.method:
            for layer, layer_arch in arch['layer'].items():
                for blk_idx, a in enumerate(layer_arch):
                    if a == 0:
                        if layer == 'self_attn':
                            getblock(self.model, self.config)[blk_idx].skip_attn(reuse=reuse)
                        elif layer == 'mlp':
                            getblock(self.model, self.config)[blk_idx].skip_mlp(reuse=reuse)
                    elif a == 1:
                        if layer == 'self_attn':
                            getblock(self.model, self.config)[blk_idx].use_attn()
                        elif layer == 'mlp':
                            getblock(self.model, self.config)[blk_idx].use_mlp()
        # print(f'sample time : {(time() - sample_start):.2f}, device : {accelerator.device}')
        # accelerator.wait_for_everyone()
        
        return self.model
    
    # def validate_arch(self, arch):
    #     assert all([l in self.config['linear'] for l in list(arch.keys())]), f'{list(arch.keys())} are invalid'
    #     for linear, linear_bits in arch.items():
    #         assert len(linear_bits) == self.config['n_block'], f'{linear}: len(linear_bits) != n_block'
    #         _, linear = linear.split('.')
    #         assert all([b in [0, self.small_model_bits, self.large_model_bits] for b in linear_bits]), f'{linear}: {linear_bits} are not compatible with the evaluator.'

    def eval(self, accelerator, arch, metric, loss_func='cross_entropy'):
        # if metric == 'latency':
        #     measure_latency(model=self.sample(arch))
        if metric == 'ppl':
            loaders = self.test_loaders
        elif metric == 'loss':
            loaders = self.train_loaders
        else:
            raise NotImplementedError(f"metric should be 'ppl' or 'loss', not {metric}")
        metric_list = dict()
        for dataset, loader in loaders.items():
            metric_list[dataset] = eval_metric(model=self.sample(arch), accelerator=accelerator, metric=metric, loader=loader, seqlen=self.seqlen, loss_func=loss_func, dense_logits_list=self.dense_logits[dataset])
        complexity = get_net_info(arch, self.config, self.latency_table)
        torch.cuda.empty_cache()
        return metric_list, complexity
    
    def remove_linears(self, model, config):
        for blk in getblock(model, config):
            for linear_group in config['linear']:
                for linear in linear_group.split(','):
                    delsubattr(blk, linear)
        gc.collect()
        torch.cuda.empty_cache()

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


        # elif 'gptq' in method:
        #     self.quant_models = [GPTQModel.from_quantized(path, device_map=device_map, backend=get_backend('AUTO')).model for path in quant_model_paths]
        #     self.quant_model_bits = quant_model_bits
        #     self.model = deepcopy(self.quant_models[np.argmax(quant_model_bits)]).to(accelerator.device)
        #     self.remove_linears(self.model, config)

        # elif 'owq' in method:
        #     self.quant_models = [load_model(model_id, path, device=device_map) for path in quant_model_paths]
        #     self.quant_model_bits = quant_model_bits
        #     self.model = deepcopy(self.quant_models[np.argmax(quant_model_bits)]).to(accelerator.device)
        #     self.remove_linears(self.model, config)

        # elif 'awq' in method :
        #     with accelerator.main_process_first():
        #         self.model = AutoModelForCausalLM.from_pretrained(quant_model_paths[np.argmax(quant_model_bits)], torch_dtype='auto', device_map=device_map)
        #         self.remove_linears(self.model, config)
        #         self.quant_models = [AutoModelForCausalLM.from_pretrained(p, torch_dtype='auto', device_map=device_map) for p in quant_model_paths]
        #     self.quant_model_bits = quant_model_bits