import re
import math
import argparse

import numpy as np
from transformers import AutoConfig

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='meta-llama', help='model path')
parser.add_argument('--model_name', type=str, default='Llama-2-7b-hf', help='model name')
parser.add_argument('--bit', type=float, default=16.0, help='bit')
parser.add_argument('--group_size', type=int, default=128, help='group size')

args = parser.parse_args()
model_id = f'{args.model_path}/{args.model_name}'

config = AutoConfig.from_pretrained(model_id)

num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
intermediate_size = config.intermediate_size
num_attention_heads = config.num_attention_heads
num_key_value_heads = config.num_key_value_heads
vocab_size = config.vocab_size
model_bit = int(re.sub(r'[^0-9]', '', str(config.torch_dtype)))

# Output dim * input dim
query_proj_size = out_proj_size = hidden_size * hidden_size
key_proj_size = value_proj_size = hidden_size / (num_attention_heads // num_key_value_heads) * hidden_size
up_proj_size = gate_proj_size = down_proj_size = intermediate_size * hidden_size

# Mega Bytes
one_bit_size = (query_proj_size + key_proj_size + value_proj_size + out_proj_size + up_proj_size + gate_proj_size + down_proj_size) * num_layers / 8 / 1024 / 1024

# scale + zero size
if args.bit <= 4:
    if args.group_size == 1:
        query_scale = out_scale = down_scale = hidden_size
        key_scale = value_scale = hidden_size / (num_attention_heads // num_key_value_heads)
        up_scale = gate_scale = intermediate_size

        # Mega Bytes
        one_group_size = (query_scale + key_scale + value_scale + out_scale + up_scale + gate_scale + down_scale) * num_layers * model_bit / 8 / 1024 / 1024
    elif args.group_size == 128:
        query_scale = out_scale = hidden_size * (hidden_size / 128)
        key_scale = value_scale = hidden_size / (num_attention_heads // num_key_value_heads) * (hidden_size / 128)
        up_scale = gate_scale = down_scale = intermediate_size * (hidden_size / 128)

        # Mega Bytes
        one_group_size = (query_scale + key_scale + value_scale + out_scale + up_scale + gate_scale + down_scale) * num_layers * model_bit / 8 / 1024 / 1024

    one_group_zero_size = one_group_size * 2
else:
    one_group_zero_size = 0

# Mega Bytes
embed_head_size = 2 * hidden_size * vocab_size * model_bit / 8 / 1024 / 1024

# Mega Bytes
norm_size = (2 * hidden_size * num_layers + hidden_size) * model_bit / 8 / 1024 / 1024

print(one_bit_size, one_group_zero_size, embed_head_size, norm_size)

if args.bit <= 4:
    for bit in np.linspace(2, 4, 41):
        expected_size = one_bit_size * bit + one_group_zero_size + embed_head_size + norm_size
    
        print(bit, math.ceil(expected_size))

# print(one_bit_size, one_group_zero_size, embed_head_size, norm_size)
# print(expected_size)
else:
    expected_size = one_bit_size * args.bit + one_group_zero_size + embed_head_size + norm_size
    print(math.ceil(expected_size))