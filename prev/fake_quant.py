import os
import json
import argparse
import numpy as np
# from evaluator import LlamaEvaluator
from transformers import AutoModelForCausalLM
from utils.eval_utils import eval_metric
from utils.data_utils import get_loader
from utils.latency_utils import test_latency
from tqdm import tqdm

import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    t = t.reshape(t_shape)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    # t = t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    # t = t.reshape(t_shape)
    return t

@torch.no_grad()
def quantize_activation_per_tensor_absmax_la(t, n_bits=8):
    t_shape = t.shape
    t_ = t.view(-1, t_shape[-1]).clone()

    act_channel_scales = t_.abs().max(dim=0, keepdim=True)[0] #
    smooth_scales = (act_channel_scales / torch.log2(2 + act_channel_scales))

    t_.div_(smooth_scales)

    act_scales = t_.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    act_scales.clamp_(min=1e-5).div_(q_max)
    t_.div_(act_scales).round_().mul_(act_scales)
    t_ = t_.reshape(t_shape)

    return t_, smooth_scales

@torch.no_grad()
def quantize_activation_per_tensor_absmax_smooth(t, weight, w_bits=4, n_bits=8, alpha=0.5):
    t_shape = t.shape
    t_ = t.view(-1, t_shape[-1]).clone()

    # import pdb; pdb.set_trace()
    act_channel_scales = t_.abs().max(dim=0, keepdim=True)[0] #
    weight_scales = weight.abs().max(dim=-1, keepdim=True)[0].to(act_channel_scales.device)
    q_max = 2 ** (w_bits - 1) - 1
    weight_scales.clamp_(min=1e-5).div_(q_max)
    smooth_scales = (act_channel_scales.pow(alpha) / weight_scales.max().pow(1 - alpha)).clamp(min=1e-5)
    # import pdb; pdb.set_trace()

    t_.div_(smooth_scales)

    act_scales = t_.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    act_scales.clamp_(min=1e-5).div_(q_max)
    t_.div_(act_scales).round_().mul_(act_scales)
    t_ = t_.reshape(t_shape)

    return t_, smooth_scales

class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        w_bits=4,
        a_bits=8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bits)
        elif act_quant == "per_tensor":
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bits)
        elif act_quant == 'per_tensor_la':
            self.act_quant = partial(quantize_activation_per_tensor_absmax_la, n_bits=a_bits)
            self.weight_quant = partial(quantize_weight_per_channel_absmax, n_bits=w_bits)
        elif act_quant == 'per_tensor_smooth':
            self.act_quant = partial(quantize_activation_per_tensor_absmax_smooth, w_bits=w_bits, n_bits=a_bits)
            self.weight_quant = partial(quantize_weight_per_channel_absmax, n_bits=w_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")
        self.act_quant_name = act_quant

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    # @torch.no_grad()
    # def forward(self, x):
    #     q_x = self.act_quant(x)
    #     y = torch.functional.F.linear(q_x, self.weight, self.bias)
    #     q_y = self.output_quant(y)
    #     return q_y

    @torch.no_grad()
    def forward(self, x):
        if self.act_quant_name in ['per_tensor_smooth', 'per_tensor_la']:
            if self.act_quant_name == 'per_tensor_la':
                q_x, smooth_scales = self.act_quant(x)
            elif self.act_quant_name == 'per_tensor_smooth' :
                q_x, smooth_scales = self.act_quant(x, self.weight)

            weight = self.weight * smooth_scales
            weight = self.weight_quant(weight)
            y = torch.functional.F.linear(q_x, weight, self.bias)
            q_y = self.output_quant(y)
        else:
            q_x = self.act_quant(x)
            y = torch.functional.F.linear(q_x, self.weight, self.bias)
            q_y = self.output_quant(y)

        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, w_bits=8, a_bits=8
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            w_bits=w_bits,
            a_bits=a_bits
        )
        if 'smooth' in act_quant or 'la' in act_quant:
            new_module.weight = module.weight
        elif weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bits
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bits
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True, w_bits=8, a_bits=8
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in tqdm(model.model.named_modules(), desc='Quantizing Model'):
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, w_bits=8, a_bits=8
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in tqdm(model.model.named_modules(), desc='Quantizing Model'):
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            # m.down_proj = W8A8Linear.from_float(
            #     m.down_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            # )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant='per_token', w_bits=w_bits, a_bits=a_bits
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            # m.o_proj = W8A8Linear.from_float(
            #     m.o_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            # )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant='per_tensor', w_bits=w_bits, a_bits=a_bits
            )
    import pdb; pdb.set_trace()
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, w_bits=8, a_bits=8
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            m.w2 = W8A8Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            m.w3 = W8A8Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, w_bits=8, a_bits=8
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                w_bits=w_bits,
                a_bits=a_bits
            )
            m.dense = W8A8Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, w_bits=w_bits, a_bits=a_bits
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, w_bits=8, a_bits=8
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            w_bits=w_bits,
            a_bits=a_bits
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            w_bits=w_bits,
            a_bits=a_bits
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            w_bits=w_bits,
            a_bits=a_bits
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            w_bits=w_bits,
            a_bits=a_bits
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
def main(args):
    print(args)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto')
    device = next(model.parameters()).device
    model = quantize_model(model, act_quant=args.act_quant, w_bits=args.w_bits, a_bits=args.a_bits)
    loader = get_loader(args.dataset, model=args.model_name, train=False, seqlen=args.seqlen)
    ppl = eval_metric(model=model, metric='ppl', loader=loader, device=device, seqlen=args.seqlen)
    # latency = test_latency(model, True, device)
    # print(f'ppl : {ppl}, latency : {latency}')
    print(f'ppl : {ppl}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    # parser.add_argument('--config', type=str, default='config/llama.json',
    #                     help='')
    # parser.add_argument('--large_model_path', type=str, default='',
    #                     help='file path to supernet weights')
    # parser.add_argument('--large_model_bits', type=int, default=4,
    #                     help='test batch size for inference')
    # parser.add_argument('--small_model_path', type=str, default='',
    #                     help='file path to supernet weights')
    # parser.add_argument('--small_model_bits', type=int, default=2,
    #                     help='test batch size for inference')
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2', 
                        help='linear list not to replace')
    parser.add_argument('--act_quant', type=str, default='per_token',
                        help='')
    parser.add_argument('--w_bits', type=int, default=8,
                        help='')
    parser.add_argument('--a_bits', type=int, default=8,
                        help='')
    # parser.add_argument('--greedy_search_result_path', type=str, default='',
    #                     help='')
    # parser.add_argument('--only_front', type=bool, default=True,
    #                     help='')
    # parser.add_argument('--results_file', type=str, default='results.txt',
    #                     help='')
    # parser.add_argument('--results_csv_file', type=str, default='results.csv',
    #                     help='')
    # parser.add_argument('--target_bits_range', type=float, nargs='+', default=[],
    #                     help='')

    cfgs = parser.parse_args()
    main(cfgs)
