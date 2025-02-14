"""
알고리즘 돌려서 ppl이랑 재는 버전
"""

import os
import gc
import json
import argparse
import torch
import numpy as np

from evaluator import LlamaEvaluator
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer
from transformers import AutoModelForCausalLM

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main(args, arch):
    print(args)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    model_id = f'{args.model_path}/{args.model_name}'
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        outlier=torch.load(args.outlier_path) if args.do_owq else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.eval_datasets,
    )

    del evaluator.model
    torch.cuda.empty_cache()
    gc.collect()

    # model = evaluator.model
    # model = model.to('cuda')


    ## customizing
    from autoquant.autoquant.algorithm import get_quantized_model
    tokenizer = get_tokenizer(model_id)
    global field

    ppl_list = []
    bits_list = []

    result = {}

    print("Get quantized model")
    """
    실험 끝나면 do clip asym True로 바꿀 것 !!!!!!!
    """

    # for multi gpu
    # max_memory={0: "40GB", 1: "40GB", 2: "40GB", 3: "40GB"},

    method = get_quantized_model(args.method, arch, model_id, 'cuda', do_prune = args.do_prune, do_owq = args.do_owq, owq_path = torch.load(args.outlier_path) if args.do_owq else None, 
                                 group_size = args.group_size, do_clip_asym = args.do_clip_asym)
    model = method.model
    model = model.to('cuda')

    print("Evaluate")
    metric, complexity = evaluator.eval_woo(arch=arch, model = model, metric='ppl', accelerator=accelerator)

    ppl_list.append({d: metric[d] for d in args.eval_datasets})
    bits_list.append(complexity['bits'])
    print(f'bits : {complexity["bits"]}, ppl : {metric}')

    result['algorithm'] = args.method
    result['bits'] = complexity['bits']

    if 'wikitext2' in args.eval_datasets:
        result['wikitext2'] = metric['wikitext2']
    if 'c4' in args.eval_datasets:
        result['c4'] = metric['c4']
    
    if args.zeroshot:
        print("Evaluate zeroshot")
        results = eval_zeroshot(model, tokenizer=tokenizer, batch_size=32, task_list=['arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'winogrande', 'boolq'])
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

    print(args)
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
    parser.add_argument('--method', type=str, default='awq',
                        help='')
    
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='')
    
    parser.add_argument('--arch_path', type=str, default=None)
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--zeroshot', action='store_true', help='Whether to evaluate zeroshot')
    parser.add_argument('--output_path', type=str, default=None,
                        help='')
    
    parser.add_argument('--target_bits', type=int, nargs='+', default=[],
                        help='')
    parser.add_argument('--group_size', type=int, default=128,
                        help='')
    
    parser.add_argument('--do_prune', action='store_true', help='Whether to use pruning')
    parser.add_argument('--do_owq', action='store_true', help='Whether to use owq')
    parser.add_argument('--do_clip_asym', action='store_true', help='Whether to clip asym')
    parser.set_defaults(do_clip_asym=True)

    parser.add_argument('--do_clip_sym', action='store_true', help='Whether to clip sym')
    parser.add_argument('--half', action='store_true', help='test arch half')
    
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')

    cfgs = parser.parse_args()

    if cfgs.do_clip_sym:
        cfgs.do_clip_asym = False

    ## customizing
    global field
    
    if cfgs.zeroshot:
        field = ['algorithm', 'bits', *cfgs.eval_datasets, 'arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'winogrande', 'boolq', 'avg']
    else:
        field = ['algorithm', 'bits', *cfgs.eval_datasets]

    assert 'llama' in cfgs.model_name.lower() , 'not supported model'
    linears = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']

    if cfgs.output_path is not None:
        # if os.path.exists(cfgs.output_path) == False:
            # os.makedirs(cfgs.output_path, exist_ok=True)

        with open(cfgs.output_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=field)
            writer.writeheader()

    if '7b' in cfgs.model_name:
        layer_len = 32
    elif '13b' in cfgs.model_name:
        layer_len = 40
    elif '70b' in cfgs.model_name:
        layer_len = 80
    else:
        assert False, 'not supported model'

    if cfgs.arch_path is not None:
        with open(cfgs.arch_path, 'r') as f:
            archs = json.load(f)['archive']

            len_archs = len(archs)

            for i, arch in enumerate(archs):
                if cfgs.half:
                    if i < len_archs // 2:
                        continue

                arch = {'linear' : arch['arch'] if type(arch) == dict else arch[0]['linear'], 
                        'layer' : {'self_attn' : [1] * layer_len, 'mlp' : [1] * layer_len}}

                print(arch)
                main(cfgs, arch)

    else:
        for target_bit in cfgs.target_bits:
            arch = {'linear' : {linear : [target_bit] * layer_len for linear in linears},
                    'layer' : {'self_attn' : [1] * layer_len, 'mlp' : [1] * layer_len}}
            
            print(arch)
            main(cfgs, arch)