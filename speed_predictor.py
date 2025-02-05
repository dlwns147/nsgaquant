import os
import json
import time
import contextlib
import random
import logging
import gc
import argparse
import math

from pathlib import Path
from typing import Optional, Tuple
from glob import glob
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaForCausalLM

from hqq.utils.patching_woo import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.autogptq import GPTQLinear
from hqq.backends.bitblas import HQQLinearBitBlas
from hqq.core.quantize import HQQLinear

from monkeypatch.ftllama_modeling import convert_model_to_ft
from monkeypatch.ftllama_generate import replace_generate_functions

from benchmark_inference_latency import profile_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['HF_HOME']='/SSD/Woo'

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = None

args = None
# size = [(1, 64), (1, 128)]
# sizes = [(1, 2048)]
sizes = [(1, 128)]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def calc_adjust_proxy_speed(proxy, layer_length, arch, sizes, common_bit = None):
    """
    proxy : token/s가 저장되어 있는 dictionary의 주소
    layer_length : base model의 layer들 개수
    arch : architecture
    size : [(batch_size, seq_len), ...]
    common_bit : 하나의 bit가 모델 전체에 공통으로 적용되는 경우, None이면 각 layer마다 bit가 다름
    """
    result = {}

    base_proxy_speed_predict = calc_proxy_speed(proxy, layer_length, arch, sizes, common_bit)

    for batch_size, seq_len in sizes:
        adjusted_speed = base_proxy_speed_predict[f'{batch_size}.{seq_len}']['median'] + proxy['deviation'][f'{batch_size}.{seq_len}']['median']

        result[f'{batch_size}.{seq_len}'] = adjusted_speed

    return result


def calc_proxy_speed(proxy, layer_length, arch, sizes, common_bit = None):
    base_proxy_speed_predict = {}
    for batch_size, seq_len in sizes:
        base_proxy_median_speed = proxy['base'][f'{batch_size}.{seq_len}']['median']
        base_proxy_mean_speed = proxy['base'][f'{batch_size}.{seq_len}']['mean']
        proxy_median_sum = 0
        proxy_mean_sum = 0

        for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
            module, linear = name.split('.')
            for idx in range(layer_length):
                if common_bit is  None:
                    bit = int(arch[name][idx])
                else:
                    bit = common_bit
                
                proxy_median_sum += (proxy[name][str(bit)][f'{batch_size}.{seq_len}']['median'] - base_proxy_median_speed)
                proxy_mean_sum += (proxy[name][str(bit)][f'{batch_size}.{seq_len}']['mean'] - base_proxy_mean_speed)

        proxy_median_sum = proxy_median_sum / layer_length
        proxy_mean_sum = proxy_mean_sum / layer_length

        base_proxy_speed_predict[f'{batch_size}.{seq_len}'] = {}
        base_proxy_speed_predict[f'{batch_size}.{seq_len}']['median'] = base_proxy_median_speed + proxy_median_sum
        base_proxy_speed_predict[f'{batch_size}.{seq_len}']['mean'] = base_proxy_mean_speed + proxy_mean_sum

    return base_proxy_speed_predict


def get_model():
    global model_name, args
    model_name = args.model_name_or_path
    model_path, model_id = model_name.split('/')

    int2_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis')
    int3_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis')
    int4_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis')

    assert args.backend_3bit == 'gptq', '3bit only support gptq backend'
    prepare_for_inference(int2_model, backend = args.backend_2bit, load_path = f"/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int3_model, backend = args.backend_3bit, load_path = f"/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int4_model, backend = args.backend_4bit, load_path = f"/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis_GPTQLinear.pt")

    int2_layers = int2_model.model.layers
    int3_layers = int3_model.model.layers
    int4_layers = int4_model.model.layers

    del int2_model
    del int3_model
    del int4_model

    cleanup()

    base_model = get_hfmodel(model_name, dtype='float16', attn_implementation='ft' if args.use_ft else 'hf')
    base_model.eval()
    
    base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
    base_model.config.use_cache = True
    base_model.generation_config.use_cache = True

    cleanup()

    return base_model, int2_layers, int3_layers, int4_layers


def get_proxy():
    global args
    model, int2_layers, int3_layers, int4_layers = get_model()

    model_layers = model.model.layers
    proxy_model = model.to(default_device)     ## We expect the default_devide to be 'cuda'

    proxy_result = {}

    cleanup()
    
    print("Get Base(FP16) Speed...")
    device_sync(default_device)
    start = time.perf_counter()
    token_per_second = profile_model(proxy_model, sizes = sizes, generation = True)
    device_sync(default_device)
    end = time.perf_counter()

    proxy_result['base'] = token_per_second

    print("Get Proxy...")
    for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
        module, linear = name.split('.')

        proxy_result[name] = {}
        
        ori_linear = []
        for i in range(len(model_layers)):
            ori_linear.append(getattr(getattr(proxy_model.model.layers[i], module), linear))

        for bit in [2, 3, 4]:
            layers = (None, None, int2_layers, int3_layers, int4_layers)[bit]

            for i in range(len(model_layers)):
                source = getattr(getattr(layers[i], module), linear)

                setattr(getattr(proxy_model.model.layers[i], module), linear, source.to(default_device))

            device_sync(default_device)
            start = time.perf_counter()
            token_per_second = profile_model(proxy_model, sizes = sizes, generation = True)
            device_sync(default_device)
            end = time.perf_counter()

            proxy_result[name][str(bit)] = token_per_second

            print(f"{name}, {bit} is done")

        for i in range(len(model_layers)):
            setattr(getattr(proxy_model.model.layers[i], module), linear, ori_linear[i])

        cleanup()


    print("Get deviation...")
    layer_length = len(proxy_model.model.layers)
    proxy_result["quantization"] = {}
    proxy_result["deviation"] = {f'{batch_size}.{seq_len}': {'median': 0, 'mean': 0} for batch_size, seq_len in sizes}

    for bit in [2, 3, 4]:
        quantized_layers = (None, None, int2_layers, int3_layers, int4_layers)[bit]

        for layer_idx, layer in enumerate(layers):
            named_linears = get_named_linears(layer)
            for name in named_linears:
                module, linear = name.split('.')

                source = getattr(getattr(quantized_layers[layer_idx], module), linear)

                setattr(getattr(model.model.layers[layer_idx], module), linear, source.to('cuda'))
        
        device_sync(default_device)
        start = time.perf_counter()
        token_per_second = profile_model(proxy_model, sizes = sizes, generation = True)
        device_sync(default_device)
        end = time.perf_counter()

        proxy_result["quantization"][str(bit)] = token_per_second

        for key, value in token_per_second.items():
            proxy_result["deviation"][key]['median'] += value['median']
            proxy_result["deviation"][key]['mean'] += value['mean']

        ## TODO : proxy 계산해서 deviation 계산
        # import code; code.interact("base proxy speed predict", local=dict(globals(), **locals()))
        base_proxy_speed_predict = calc_proxy_speed(proxy_result, layer_length, None, sizes, bit)
        for key, value in base_proxy_speed_predict.items():
            proxy_result["deviation"][key]['median'] -= value['median']
            proxy_result["deviation"][key]['mean'] -= value['mean']

    # import code; code.interact("proxy result", local=dict(globals(), **locals()))
    for key in proxy_result["deviation"].keys():
        proxy_result["deviation"][key]['median'] /= 3
        proxy_result["deviation"][key]['mean'] /= 3

    proxy_model = proxy_model.to('cpu')
    del proxy_model
    cleanup()

    model = model.to(default_device)

    return proxy_result


def device_sync(default_device):
    if "cuda" in default_device:
        torch.cuda.synchronize(default_device)
    elif ("cpu" in default_device) or ("mps" in default_device):
        pass
    else:
        print(f"device={default_device} is not yet suppported")


def get_named_linears(module, specific_cls=None):
    if specific_cls is None:
        return {name: m for name, m in module.named_modules() if (isinstance(m, nn.Linear) 
                                                              or isinstance(m, GPTQLinear)
                                                              or isinstance(m, HQQLinearBitBlas))
                                                              or isinstance(m, HQQLinear)}
    else:
        return {name: m for name, m in module.named_modules() if isinstance(m, specific_cls)}


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def _profile_model(model, input_data, attention_mask, min_warmup_time=1.0, generation = True):
    model = model.cuda()
    model.eval()
    
    def get_runtime(num_repeats=1):
        times = []
        
        for _ in range(num_repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_data, attention_mask = attention_mask)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)      ## ms 단위로 변환
        
        return torch.tensor(times)


    def get_token_per_second(seq_length = 128, num_repeats=1):
        times = []
        
        for _ in range(num_repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.generate(input_data, 
                            min_new_tokens = seq_length,
                            max_new_tokens = seq_length,
                            do_sample=False,
                            num_beams=1,
                            attention_mask = attention_mask)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)      ## ms 단위로 변환
        
        return torch.tensor(times)

    # Warmup phase
    with torch.no_grad():
        if generation:
            seq_length = input_data.size(1)
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < min_warmup_time:
                get_token_per_second(seq_length)

            # Measurement phase
            warmup_runtime = get_token_per_second(seq_length).mean().item()
            num_repeats = max(1, int(1000 / warmup_runtime))       ## 10초 동안 몇 번 반복할 수 있는지 계산, 최소 1번
            times = get_token_per_second(seq_length, num_repeats)
            # times = get_token_per_second(seq_length, 10)
            times = seq_length / (times / 1000)     ## tokens per second

            return {
                'mean': times.mean().item(),
                # 'std': times.std().item(),
                'median': times.median().item()
            }


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
    
    kwargs.pop('attn_implementation')    
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name_or_path', type=str, help='model path')
    parser.add_argument('gpu_name', type=str, help='gpu name')
    parser.add_argument('--use_ft', action='store_true', help='use faster transformer')

    parser.add_argument('--backend_2bit', type=str, choices=['gptq', 'bitblas'], help='backend for 2bit', default = 'gptq')
    parser.add_argument('--backend_3bit', type=str, choices=['gptq', 'bitblas'], help='backend for 3bit', default = 'gptq')
    parser.add_argument('--backend_4bit', type=str, choices=['gptq', 'bitblas'], help='backend for 4bit', default = 'gptq')

    parser.add_argument('--arch_path', type=str, help='arch path', default = "/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-7b-hf.json")

    global args
    args = parser.parse_args()

    model_name, model_id = args.model_name_or_path.split('/')
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    layer_length = config.num_hidden_layers

    proxy_path = f'speed_predictor_json/{model_id}_{args.gpu_name}.json'

    if os.path.exists(proxy_path):
        with open(proxy_path, 'r') as f:
            proxy = json.load(f)
    else:
        proxy = get_proxy()
        with open(proxy_path, 'w') as f:
            json.dump(proxy, f, indent = 4)

    arch_path = args.arch_path
    with open(arch_path, 'r') as f:
        data = json.load(f)
        archive = data['archive']

    for i, archs in enumerate(archive):
        print("Arch Index : ", i)

        adjusted_speed = calc_adjust_proxy_speed(proxy, layer_length, archs['arch'], sizes, None)

        print(adjusted_speed)


if __name__ == '__main__':
    main()
