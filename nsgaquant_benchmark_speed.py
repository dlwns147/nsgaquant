import os
import gc
import csv
import math
import json
import argparse
from copy import deepcopy

import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from hqq.utils.patching import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.autogptq import GPTQLinear
from hqq.backends.bitblas import HQQLinearBitBlas
from hqq.core.quantize import HQQLinear
from monkeypatch.ftllama_modeling import convert_model_to_ft
from monkeypatch.ftllama_generate import replace_generate_functions

from benchmark.benchmark_speed import benchmark_speed

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def get_named_linears(module, specific_cls=None):
    if specific_cls is None:
        return {name: m for name, m in module.named_modules() if (isinstance(m, nn.Linear) 
                                                              or isinstance(m, GPTQLinear)
                                                              or isinstance(m, HQQLinearBitBlas)
                                                              or isinstance(m, HQQLinear))}
    else:
        return {name: m for name, m in module.named_modules() if isinstance(m, specific_cls)}
    

def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                device_map='cpu',
                trust_remote_code=False,
                **kwargs
                ):

    assert kwargs.get('attn_implementation') in ['hf', 'ft']        ## hf : huggingface, ft : faster transformer
    
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    ft = False
    if kwargs.get('attn_implementation') == 'ft':
        assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
        ft = True
    
    print('attention implementaion is :', kwargs.pop('attn_implementation'))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    if ft:
        convert_model_to_ft(model)
        replace_generate_functions()

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model


def get_memory_footprint(module: torch.nn.Module, return_buffers: bool = True) -> int:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Input must be a PyTorch Module")
    mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
        mem = mem + mem_bufs
    return mem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name_or_path', type=str, help='model path')
    parser.add_argument('--use_ft', action='store_true', help='use faster transformer')
    parser.add_argument('--use_owq', action='store_true', help='use owq')

    parser.add_argument('--backend_2bit', type=str, choices=['gptq', 'bitblas', 'gemlite', 'gptq_cuda', 'gptq_tritonv2'], help='backend for 2bit', default = 'gptq')
    parser.add_argument('--backend_3bit', type=str, choices=['gptq', 'gptq_cuda', 'gptq_tritonv2'], help='backend for 3bit', default = 'gptq')
    parser.add_argument('--backend_4bit', type=str, choices=['gptq', 'bitblas', 'gemlite', 'gptq_cuda', 'gptq_tritonv2', 'qeft'], help='backend for 4bit', default = 'gptq')

    parser.add_argument('--batch_size', type=int, help='batch size', default = 1)
    parser.add_argument('--seq_length', type=int, help='sequence length', default = 64)
    parser.add_argument('--gen_length', type=int, help='generation length', default = 128)

    parser.add_argument('--tps', action='store_true', help='token per second')
    parser.add_argument('--gemm', action='store_true', help='gemm')
    parser.add_argument('--gemv', action='store_true', help='gemv')
    parser.add_argument('--ttft', action='store_true', help='ttft')
    parser.add_argument('--memory', action='store_true', help='memory')
    parser.add_argument('--peak_memory', action='store_true', help='peak memory')

    parser.add_argument('--file_name', type=str, help='save path', default = None)

    args = parser.parse_args()

    args.use_arch = True

    global model_name
    model_name = args.model_name_or_path
    model_path, model_id = model_name.split('/')
    result = {}

    int2_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis')
    int3_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis')
    int4_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis')

    int2_model = int2_model.to(default_device)
    int3_model = int3_model.to(default_device)
    int4_model = int4_model.to(default_device)

    prepare_for_inference(int2_model, backend = args.backend_2bit, load_path = f"/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis_{args.backend_2bit.upper()}Linear.pt")
    prepare_for_inference(int3_model, backend = args.backend_3bit, load_path = f"/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis_{args.backend_3bit.upper()}Linear.pt")
    prepare_for_inference(int4_model, backend = args.backend_4bit, load_path = f"/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis_{args.backend_4bit.upper()}Linear.pt")

    int2_layers = int2_model.model.layers
    int3_layers = int3_model.model.layers
    int4_layers = int4_model.model.layers

    int2_model = int2_model.to('cpu')
    int3_model = int3_model.to('cpu')
    int4_model = int4_model.to('cpu')

    if args.use_owq:
        owq_path = f"/NAS/Woo/Automation/autoopt/kernel/outlier/{model_id}/w16_r32/outlier.pth"
        owq = torch.load(owq_path, weights_only = True) if os.path.exists(owq_path) else None

        int2_owq_model = deepcopy(int2_model)
        int3_owq_model = deepcopy(int3_model)
        int4_owq_model = deepcopy(int4_model)

        intN_models = [int2_owq_model, int3_owq_model, int4_owq_model]
        for intN_model in intN_models:
            linears = get_named_linears(intN_model, specific_cls=HQQLinear)
            for name, module in linears.items():
                module.name = name

                if 'o_proj' in name:
                    continue

                owq_layer = owq[name]
                module.do_owq = False
                module.register_buffer('oweight', torch.zeros((len(owq_layer), module.out_features), dtype=torch.float))
                module.register_buffer('outlieridx', torch.Tensor(owq_layer).to(torch.int))

        prepare_for_inference(int2_owq_model, backend = args.backend_2bit, load_path = f"/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis_{args.backend_2bit.upper()}Linear_owq.pt", owq = True)
        prepare_for_inference(int3_owq_model, backend = args.backend_3bit, load_path = f"/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis_{args.backend_3bit.upper()}Linear_owq.pt", owq = True)
        prepare_for_inference(int4_owq_model, backend = args.backend_4bit, load_path = f"/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis_{args.backend_4bit.upper()}Linear_owq.pt", owq = True)

        int2_owq_layers = int2_owq_model.model.layers
        int3_owq_layers = int3_owq_model.model.layers
        int4_owq_layers = int4_owq_model.model.layers

        del int2_owq_model
        del int3_owq_model
        del int4_owq_model

    cleanup()

    base_model = get_hfmodel(model_name, dtype='float16', attn_implementation='ft' if args.use_ft else 'hf')
    base_layers = base_model.model.layers

    base_model.eval()
    base_model = base_model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        use_fast=False,
        trust_remote_code=True
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sizes = [args.batch_size, args.seq_length, args.gen_length]
    gemm_iteration = 20
    gemv_iteration = 5 if args.gen_length < 1024 else 2

    print(f"Get Speed...")
    result['fp16'] = {}

    if args.tps:
        token_per_second = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'TPS', get_peak_memory=args.peak_memory)
        result['fp16'].update(token_per_second)
        print('Token per second : ', token_per_second)

    if args.gemm:
        gemm = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'GeMM', get_peak_memory=False)
        result['fp16'].update(gemm)
        print('GeMM : ', gemm)

    if args.gemv:
        gemv = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'GeMV', get_peak_memory=False)
        result['fp16'].update(gemv)
        print('GeMV : ', gemv)

    if args.ttft:
        ttft = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'TTFT', get_peak_memory=False)
        result['fp16'].update(ttft)
        print('TTFT : ', ttft)

    if args.memory:
        memory = get_memory_footprint(base_model) / 1024 ** 3
        result['fp16'].update({'memory' : memory})
        print(f"Base Model Memory : {memory} GB")

    if args.file_name:
        result_dir = f'benchmark/outputs'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, args.file_name)        
                       
    base_model = base_model.to('cpu')
    base_layers_length = len(base_layers)
    linears = list(get_named_linears(base_layers[0]).keys())

    if args.use_arch:
        arch_7b = '/NAS/SJ/nsgaquant/save/search/quant/2502101708_Llama-2-7b-hf_bits_loss_hqq_iter_200_234_obj_2_4.1_jsd_co_0.9_mut_0.1_wikitext2_128sample_pop_200_100_rbf/iter_200.stats'
        arch_13b = '/NAS/SJ/nsgaquant/save/search/quant/2502101858_Llama-2-13b-hf_bits_loss_hqq_iter_250_234_obj_2_4.1_jsd_co_0.9_mut_0.1_wikitext2_128sample_pop_200_100_rbf/iter_200.stats'

        with open(arch_7b, 'r') as f:
            arch_7b = json.load(f)
            candidate_7b = arch_7b['candidates']
        
        with open(arch_13b, 'r') as f:
            arch_13b = json.load(f)
            candidate_13b = arch_13b['candidates']

        candidates_7b = []
        for candidate in candidate_7b:
            if abs(candidate[-1] - 2.5) < 0.05:
                candidates_7b.append(candidate)

        candidates_bits = [np.concatenate([bit for bit in candidate[0]['linear'].values()]) for candidate in candidates_7b]
        count_4bit = [(bits == 4.0).sum() for bits in candidates_bits]
        candidate_7b = candidates_7b[np.argmax(count_4bit)]

        candidates_13b = []
        for candidate in candidate_13b:
            if abs(candidate[-1] - 2.5) < 0.05:
                candidates_13b.append(candidate)

        candidates_bits = [np.concatenate([bit for bit in candidate[0]['linear'].values()]) for candidate in candidates_13b]
        count_4bit = [(bits == 4.0).sum() for bits in candidates_bits]
        candidate_13b = candidates_13b[np.argmax(count_4bit)]
        
    for bit in [2, 3, 4]:
        arch = {linear : [bit] * base_layers_length for linear in linears}

        if args.use_arch:
            if '7b' in args.model_name_or_path.lower():
                arch = candidate_7b[0]['linear']
            elif '13b' in args.model_name_or_path.lower():
                arch = candidate_13b[0]['linear']

        model = deepcopy(base_model)

        print("Replacing...")
        for layer_idx, layer in enumerate(base_layers):
            named_linears = get_named_linears(layer)
            for name in named_linears:
                module, linear = name.split('.')

                if math.isclose(arch[name][layer_idx], 2):
                    source = getattr(getattr(int2_layers[layer_idx], module), linear)
                elif math.isclose(arch[name][layer_idx], 3):
                    source = getattr(getattr(int3_layers[layer_idx], module), linear)
                elif math.isclose(arch[name][layer_idx], 4):
                    source = getattr(getattr(int4_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 2):
                    source = getattr(getattr(int2_owq_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 3):
                    source = getattr(getattr(int3_owq_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 4):
                    source = getattr(getattr(int4_owq_layers[layer_idx], module), linear)
                else:
                    raise ValueError(f'bit should be 2, 3, 4, but got {arch[name][layer_idx]}')

                if hasattr(getattr(model.model.layers[layer_idx], module), linear):
                    delattr(getattr(model.model.layers[layer_idx], module), linear)

                setattr(getattr(model.model.layers[layer_idx], module), linear, source)

        cleanup()

        model.eval()
        model = model.to('cuda')

        print(f"Get Speed...")
        result[f'{bit}bit'] = {}

        if 'gemlite' in [args.backend_2bit, args.backend_4bit] and bit in [2, 4]:
            import gemlite
            gemlite.core.GEMLITE_TRITON_RESTRICT_M = True #Restrict the batch-size to powers of 2 if True
            if os.path.exists(f'gemlite_config_{bit}.json'):
                gemlite.core.GemLiteLinear.load_config(f'gemlite_config_{bit}.json') #Load
            else:
                print("optimize gemlite kernel")
                token_per_second = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = 1, sizes = sizes, mode = 'TPS', get_peak_memory=False)
                gemlite.core.GemLiteLinear.cache_config(f'gemlite_config_{bit}.json') #Cache- run this over multiple batch-sizes
            # import code; code.interact('gemlite triton cache', local = dict(globals(), **locals()))

        if args.tps:
            token_per_second = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'TPS', get_peak_memory=args.peak_memory)
            result[f'{bit}bit'].update(token_per_second)
            print('Token per second : ', token_per_second)

        if args.gemm:
            gemm = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'GeMM', get_peak_memory=False)
            result[f'{bit}bit'].update(gemm)
            print('GeMM : ', gemm)
        
        if args.gemv:
            gemv = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'GeMV', get_peak_memory=False)
            result[f'{bit}bit'].update(gemv)
            print('GeMV : ', gemv)

        if args.ttft:
            ttft = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'TTFT', get_peak_memory=False)
            result[f'{bit}bit'].update(ttft)
            print('TTFT : ', ttft)

        if args.memory:
            memory = get_memory_footprint(model) / 1024 ** 3
            result[f'{bit}bit'].update({'memory' : memory})
            print(f"Quantized Model Memory : {memory} GB")

        model = model.cpu()
        del model
        cleanup()

    if args.file_name:
        with open(result_path, 'w') as f:
            result.update({'args' : vars(args)})
            result.update({'unit' : {'tps' : 'tokens/second', 'gemm' : 'tokens/second', 'gemv' : 'tokens/second', 'ttft' : 'latency(ms)', 'peak_memory' : 'GB'}})
            json.dump(result, f, indent=4)

        field = list(result['fp16'].keys())
        fp16_value = list(result['fp16'].values())
        int2_value = list(result['2bit'].values())
        int3_value = list(result['3bit'].values())
        int4_value = list(result['4bit'].values())

        fp16_value = [list(v.values())[0] if isinstance(v, dict) else v for v in fp16_value]
        int2_value = [list(v.values())[0] if isinstance(v, dict) else v for v in int2_value]
        int3_value = [list(v.values())[0] if isinstance(v, dict) else v for v in int3_value]
        int4_value = [list(v.values())[0] if isinstance(v, dict) else v for v in int4_value]

        with open(result_path.replace('.json', '.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(field)
            writer.writerow(fp16_value)
            writer.writerow(int2_value)
            writer.writerow(int3_value)
            writer.writerow(int4_value)


if __name__ == '__main__':
    main()
