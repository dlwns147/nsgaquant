import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from autoquant.autoquant.method.base import BASE
from autoquant.autoquant.model.skip_llama import LlamaDecoderSkipLayer

import gc
import tqdm
import math
import functools

from collections import defaultdict


class AWQ(BASE):
    def __init__(self, model_name, config, dev, arch, do_prune = False, do_owq = False, owq = None, **kwargs):
        super().__init__(model_name, config, dev, arch, do_prune, do_owq, owq)
        self.method = 'hqq'


    @torch.no_grad()
    def run(self, calib_dataset):
        pass