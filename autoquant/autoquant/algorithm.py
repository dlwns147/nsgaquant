import argparse
import json

from autoquant.autoquant.method.awq import AWQ
from autoquant.autoquant.method.gptq import GPTQ

import torch
from lm_eval.models import huggingface
from lm_eval import evaluator

from init_seed import init_seed
from manage_json import *


args2method = {
    'gptq': GPTQ,
    'awq': AWQ
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('method', type=str, choices=['gptq', 'awq'], help='Method to use')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Name of the model')
    parser.add_argument('--arch_path', type=str, required=True, help='Path to the architecture file')
    parser.add_argument('--arch_idx', type=int, default=0, help='Index of the architecture')
    # parser.add_argument('--eval', action='store_true', help='Flag to run evaluation')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--result_save_name', type=str, help='Name of the result file directory')

    parser.add_argument('--do_owq', action='store_true', help='Whether to use OWQ')
    parser.add_argument('--owq_path', type=str, default='/NAS/SJ/nsgaquant/outlier/Llama-2-7b-hf/w16_r32/outlier.pth', help='Path to the OWQ file')

    parser.add_argument('--do_prune', action='store_true', help='Whether to use pruning')

    ## multi-process
    parser.add_argument('--multi_process', action='store_true', help='Whether to use multi-process')
    parser.add_argument('--start_gpu_id', type=int, default=0, help='Start GPU ID')
    parser.add_argument('--end_gpu_id', type=int, default=3, help='End GPU ID')


    return parser.parse_args()


def get_quantized_model(method, arch, model_name, dev, do_prune = False, do_owq = False, owq_path = None, **kwargs):
    method = args2method[method](model_name = model_name, config = None, dev = dev, arch = arch, do_prune = do_prune, do_owq = do_owq, owq = owq_path, **kwargs)
    group_size = kwargs.get('group_size', 128)
    method.set_group_size(group_size)

    if do_prune:
        print('Pruning the model')
        method.prune_model()
        
    method.run(method.get_calib_dataset())

    return method
    

def auto_quant():
    args = get_args()

    print(args)

    with open(args.arch_path, 'r') as f:
        nsga_data = json.load(f)
        len_archs = len(nsga_data['archive'])

    if args.result_save_name:
        result_ppl_path = init_json(args = args, save_path = '/NAS/Woo/Automation/autoopt/result', save_name = args.result_save_name, model_name = args.model_name.split("/")[-1], algorithm = True, dataset = 'wikitext2', metric = 'ppl')
        result_sample_ppl_path = init_json(args = args, save_path = '/NAS/Woo/Automation/autoopt/result', save_name = args.result_save_name, model_name = args.model_name.split("/")[-1], algorithm = True, dataset = 'wikitext2', metric = 'sample_ppl')

    init_seed(args.seed)

    if args.multi_process:
        cur_arch_idx = 0

        ## TODO: 데이터를 미리 다운 받아놓기
        
        while cur_arch_idx < len_archs:
            for gpu_id in range(args.start_gpu_id, args.end_gpu_id + 1):
                arch = nsga_data['archive'][cur_arch_idx][0]['linear']
                method = get_quantized_model(args.method, arch, args.model_name, f'cuda:{gpu_id}', args.do_prune, args.do_owq, args.owq_path)

                cur_arch_idx += 1

    else:
        assert args.arch_idx != -1, 'Please specify the architecture index'

        arch = nsga_data['archive'][args.arch_idx][0]['linear']
        method = get_quantized_model(args.method, arch, args.model_name, 'cuda:0', args.do_prune, args.do_owq, args.owq_path)

    # leaderboard = 'hellaswag'
    # method.model.to('cuda')
    # method.model.seqlen = 2048
    # method.model.eval()
    # lm = huggingface.HFLM(pretrained = method.model, tokenizer = method.tokenizer, max_length=2048, device='cuda', dtype=torch.float16, batch_size='auto')
    # result = evaluator.simple_evaluate(model = lm, tasks = [leaderboard], device='cuda')
    # print(result['results'][leaderboard])

    # if args.eval:
    #     from get_eval import get_eval_wData
    #     from get_eval import get_eval
    #     metric_ppl = get_eval(method.model, args.model_name)
    #     if args.result_save_name:
    #         write_json(result_ppl_path, [arch, nsga_data['archive'][args.arch_idx][1], {'wikitext2' : metric_ppl['ppl']['wikitext2']}])
    #         write_json(result_sample_ppl_path, [arch, nsga_data['archive'][args.arch_idx][1], {'wikitext2' : metric_ppl['sample_ppl']['wikitext2']}])


if __name__ == '__main__':
    auto_quant()