import random

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from autoquant.autoquant.model import skip_llama

def get_hfmodel(model_name_or_path: str,
                device_map='auto',
                dtype='auto',
                trust_remote_code=False,
                use_cache=False,
                **kwargs
                ):

    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        use_cache=use_cache,
        **kwargs
    )
    model.config.use_cache = use_cache

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model


def get_awq_calib_dataset(data="pileval", tokenizer=None, n_samples=128, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def get_gptq_calib_dataset(data = "c4", tokenizer = None, n_samples = 128, seed = 0, seqlen = 2048):
    if data == "c4":
        traindata = load_dataset(
        # 'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    else:
        raise NotImplementedError

    random.seed(seed)
    trainloader = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


class BASE:
    def __init__(self, model_name, dev, arch, do_prune = False, do_owq = False, owq = None, dtype = 'auto', device_map = 'auto', **kwargs):
        self.model_name = model_name
        self.dev = dev
        self.arch = arch

        self.do_prune = do_prune
        self.do_owq = do_owq
        self.owq = None
        if do_owq:
            if isinstance(owq, str):
                self.owq = torch.load(owq)
            else:
                self.owq = owq

        self.model = get_hfmodel(model_name, dtype = dtype, device_map = device_map, **kwargs)
        self.model.eval()
        self.model.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=kwargs.get('cache_dir', None))

        self.group_size = self.set_group_size(128)


    def prune_model(self):
        self.model = skip_llama.block_replace(self.model)

        # "layer": {"self_attn": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    # "mlp": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

        for block_name in self.arch['layer'].keys():
            if block_name == 'self_attn':
                skip_block = skip_llama.skip_attn
            elif block_name == 'mlp':
                skip_block = skip_llama.skip_mlp

            for idx, use in enumerate(self.arch['layer'][block_name]):
                if use == 0:        ## skip the block
                    skip_block(self.model, idx, reuse=False)


    def get_calib_dataset(self):
        if self.method == 'gptq':
            self.get_gptq_calib_dataset()
        elif self.method == 'awq':
            self.get_awq_calib_dataset()

        
    def set_group_size(self, group_size):
        self.group_size = group_size

    
    def get_gptq_calib_dataset(self, calib_data='c4', n_samples=128, seqlen=2049, seed=0):
        samples = get_gptq_calib_dataset(
            data=calib_data, tokenizer=self.tokenizer, n_samples=n_samples, seed=seed, seqlen=self.model.seqlen
        )
        return samples
        

    def get_awq_calib_dataset(self, calib_data = 'pileval', n_samples = 128):
        assert n_samples == 128, "n_samples must be 128"
        samples = get_awq_calib_dataset(
            data=calib_data, tokenizer=self.tokenizer, n_samples=n_samples, block_size=512
        )
        samples = torch.cat(samples, dim=0)
        return samples


    def append_str_prefix(self, x, prefix):
        if isinstance(x, str):
            return prefix + x
        elif isinstance(x, tuple):
            return tuple([self.append_str_prefix(y, prefix) for y in x])
        elif isinstance(x, list):
            return [self.append_str_prefix(y, prefix) for y in x]
        else:
            return x
    

    @staticmethod
    def is_owq(n_bit):
        return round(n_bit) != n_bit


    @staticmethod
    def get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
    

    @staticmethod    
    def get_op_by_name(module, op_name):
        # get the op by its name relative to the module
        for name, m in module.named_modules():
            if name == op_name:
                return m
        raise ValueError(f"Cannot find op {op_name} in module {module}")


    @staticmethod
    def set_op_by_name(layer, name, new_module):
        levels = name.split(".")
        if len(levels) > 1:
            mod_ = layer
            for l_idx in range(len(levels) - 1):
                if levels[l_idx].isdigit():
                    mod_ = mod_[int(levels[l_idx])]
                else:
                    mod_ = getattr(mod_, levels[l_idx])
            setattr(mod_, levels[-1], new_module)
        else:
            setattr(layer, name, new_module)


    @staticmethod
    def get_op_name(module, op):
        # get the name of the op relative to the module
        for name, m in module.named_modules():
            if m is op:
                return name
        raise ValueError(f"Cannot find op {op} in module {module}")


    