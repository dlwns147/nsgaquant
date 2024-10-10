import fire
import copy
from time import time
from tqdm import tqdm
import os
import csv

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from copy import deepcopy
import gc
import json

from utils.eval_utils import eval_metric, load_and_eval_ppl
from utils.data_utils import get_loader
from utils.func_utils import getsubattr, setsubattr, getblock, get_net_info

def greedy_search_linear_reverse(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        large_model_path: str ='',
        large_model_bit: float = 16,
        seed: int = 0,
        n_sample: int = 128,
        seqlen: int = 2048,
        dataset: str = 'wikitext2',
        eval_ppl: bool = True,
        eval_zeroshot: bool = False,
        small_model_path: str = '',
        small_model_bit: float = 2,
        target_bit: int = 3,
        eval_ppl_iter: bool = False,
        loss_csv_file: str = '',
        ppl_csv_file: str = '',
        config: str = ''
):
    
    print(f"Model Name: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}") 
    print(f'n_sample : {n_sample}')
    print(f'eval_ppl : {eval_ppl}')
    print(f'eval_zeroshot : {eval_zeroshot}')
    print(f'large_model_path: {large_model_path}')
    print(f'large_model_bit: {large_model_bit}')
    print(f'small_model_path: {small_model_path}')
    print(f'small_model_bit: {small_model_bit}')
    print(f'target_bit : {target_bit}')
    print(f'eval_ppl_iter : {eval_ppl_iter}')
    print(f'loss_csv_file : {loss_csv_file}')
    print(f'ppl_csv_file : {ppl_csv_file}')
    print(f'config : {config}')

    device = torch.device("cuda:0")
    with open(config, 'r') as f:
        config = json.load(f)[model_name]

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto', cache_dir=None,)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    large_model = AutoHQQHFModel.from_quantized(large_model_path).to(device)
    small_model = AutoHQQHFModel.from_quantized(small_model_path).to(device)
    print(f"Loaded Model: {model_name}")
    
    # replace
    model.config.use_cache = use_cache
    model.eval()

    train_loader = get_loader(model=model_name, name=dataset, train=True, seed=seed, n_sample=n_sample, seqlen=seqlen)
    test_loader = get_loader(model=model_name, name=dataset, train=False, seed=seed, seqlen=seqlen)
    print(f"Dataloader({dataset}) loaded.")

    alive_linear_list = []
    replaced_linear_list = []
    # model_linear_bits = []
    n_block = config['n_block']
    arch = {}
    for linear in config['linear']:
        arch[linear] = [small_model_bit for _ in range(n_block)]

    for blk_idx in range(n_block):
        for linear in config['linear']:
            alive_linear_list.append(f'{blk_idx}.{linear}')
            setsubattr(getblock(model, config, blk_idx), linear, deepcopy(getsubattr(getblock(small_model, config, blk_idx), linear)))

    # loss_list = list()
    min_ppl_iter_list = dict()
    min_loss_iter_list = dict()
    bits_list = list()
    iter_time_list = list()
    # results = dict()

    # check start time
    start_point = time()

    phase = 0
    while True:
        cur_bit = get_net_info(arch, config)['bits']
        if cur_bit >= target_bit or len(alive_linear_list) == 0:
            break
        phase += 1
        phase_start_point = time()

        min_loss = 1e99
        min_loss_blk_idx = -1
        min_loss_linear = None

        for blk_idx in range(n_block):
            for linear in config['linear']:
                key = f'{blk_idx}.{linear}'
                if key in alive_linear_list:
                    linear_start = time()
                    setsubattr(getblock(model, config, blk_idx), linear, deepcopy(getsubattr(getblock(large_model, config, blk_idx), linear)))

                    loss = eval_metric(model=model, metric='loss', loader=train_loader, device=device, seqlen=seqlen)
                    torch.cuda.empty_cache()

                    if loss < min_loss:
                        min_loss = loss
                        min_loss_blk_idx = blk_idx
                        min_loss_linear = linear

                    setsubattr(getblock(model, config, blk_idx), linear, deepcopy(getsubattr(getblock(small_model, config, blk_idx), linear)))
                    gc.collect()
                    linear_time = time() - linear_start
                    print(f"Phase {phase}, current bit: {cur_bit:.2f}, [{blk_idx}.{linear} replaced] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {min_loss_blk_idx}.{min_loss_linear}, time: {linear_time:.2f}s") 

        selected_layer = f'{min_loss_blk_idx}.{min_loss_linear}'
        alive_linear_list.remove(selected_layer)
        replaced_linear_list.append(selected_layer)
        setsubattr(getblock(model, config, min_loss_blk_idx), min_loss_linear, deepcopy(getsubattr(getblock(large_model, config, min_loss_blk_idx), min_loss_linear)))
        arch[min_loss_linear][min_loss_blk_idx] = large_model_bit
        gc.collect()

        cur_bit = get_net_info(arch, config)['bits']
        bits_list.append(cur_bit)
        min_loss_iter_list[f'{min_loss_blk_idx}.{min_loss_linear}'] = min_loss

        phase_finish_point = time()
        phase_time_elapsed = phase_finish_point - phase_start_point
        iter_time_list.append(phase_time_elapsed)

        # remove block causing the least snlls increase 
        print(f"Phase_time_elapsed (s): {phase_time_elapsed:.2f}s")
        print(f"[SELECTED linear: {min_loss_blk_idx}.{min_loss_linear}, Loss={min_loss:.3f}, Bits: {cur_bit:.3f}") 
        # print(f"Current Alive Layer List: {alive_linear_list}")
        # print(f"Current Replaced Layer List: {replaced_linear_list}")


        if loss_csv_file:
            with open(loss_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(min_loss_iter_list.keys()))
                write.writerow(bits_list)
                write.writerow(list(min_loss_iter_list.values()))
                write.writerow(iter_time_list)

        if eval_ppl_iter:
            model.config.use_cache = use_cache
            ppl = eval_metric(model=model, metric='ppl', loader=test_loader, device=device, seqlen=seqlen)
            model.config.use_cache = False
            min_ppl_iter_list[f'{min_loss_blk_idx}.{min_loss_linear}'] = ppl
            if ppl_csv_file:
                with open(ppl_csv_file, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(list(min_ppl_iter_list.keys()))
                    write.writerow(bits_list)
                    write.writerow(list(min_ppl_iter_list.values()))
                    write.writerow(iter_time_list)

    finish_point = time()
    time_elapsed = finish_point - start_point

    del large_model
    del small_model
    gc.collect()

    print(
        f"Time_Elapsed: {time_elapsed}\n"
        f"Model Name: {model_name}\n"
        f"Dataset: {dataset}\n"
        f"Seed: {seed}\n" 
        f"Alive Layer List: {alive_linear_list}, {len(alive_linear_list)}\n"
        f"Quantized Layer List: {replaced_linear_list}, {len(replaced_linear_list)}\n"
        f'n_sample : {n_sample}\n'
        f'large_model_path : {large_model_path}\n'
        f'small_model_path: {small_model_path}\n'
        f'target_bit : {target_bit}\n'
    )

    if eval_ppl:
        print(f"Starting PPL evaluation...")
        # model = block_remove(model, copy.deepcopy(removal_list))
        model.config.use_cache = use_cache

        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
        print(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        print(f"C4 PPL = {c4_ppl:.2f}")

    # if eval_zeroshot:
    #     del model
        
    #     print(f"Starting Zero-shot tasks evaluation...")
    #     if '30b' or '66b' or '70b' in model_name:
    #         parallelize = True
    #     else:
    #         parallelize = False

    #     tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']

    #     results = eval_zero_shot(model_name, skip_attn_list, skip_mlp_list, tasks, parallelize=parallelize)
    #     results = results['results']

if __name__ == "__main__":
    fire.Fire(greedy_search_linear_reverse)