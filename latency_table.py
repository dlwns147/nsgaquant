import torch
import os
import json
import argparse
import gc

from utils.eval import measure_latency
from model.skip_llama import block_replace
from transformers import AutoModelForCausalLM

def main(args):
    model_id = f'{args.model_path}/{args.model_name}'
    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)

    
    print("==================================================")
    print("Experiment Environment")
    print(f"Current GPU: {gpu_name}")
    print(f"# GPU: {str(gpu_num)}")
    print(f"Model Name: {model_id}")
    print(f"Infernce type : {'Token Generation' if args.generation else 'Prompt Processing'}")
    print("==================================================")

    model.eval()
    model = block_replace(model)
    latency_table = {}

    total_latency = measure_latency(model, generation=args.generation, device=model.device, batch_size=args.batch_size, iteration=args.iteration)
    print(f"Total Latency: {total_latency:.2f}")
    latency_table['total'] = total_latency


    if 'llama' in args.model_name.lower() :
        blocks = model.model.layers
    elif 'opt' in args.model_name.lower() :
        blocks = model.model.decoder.layers

    num_blocks = len(blocks)

    del blocks[1:]
    gc.collect()
    torch.cuda.empty_cache()

    one_block_latency = measure_latency(model, generation=args.generation, device=model.device, batch_size=args.batch_size, iteration=args.iteration)
    print(f"One block Latency: {one_block_latency:.4f}")

    blocks[0].skip_mlp()
    
    one_attn_block_latency = measure_latency(model, generation=args.generation, device=model.device, batch_size=args.batch_size, iteration=args.iteration)
    print(f"One attn block Latency: {one_attn_block_latency:.4f}")

    blocks[0].use_mlp()
    blocks[0].skip_attn()
    
    one_mlp_block_latency = measure_latency(model, generation=args.generation, device=model.device, batch_size=args.batch_size, iteration=args.iteration)
    print(f"One mlp block Latency: {one_mlp_block_latency:.4f}")

    blocks[0].skip_mlp()
    # del blocks[0]
    # gc.collect()
    # torch.cuda.empty_cache()

    etc_latency = measure_latency(model, generation=args.generation, device=model.device, batch_size=args.batch_size, iteration=args.iteration)
    print(f"Embed head Latency: {etc_latency:.4f}")

    attn_latency = one_attn_block_latency - etc_latency
    mlp_latency = one_mlp_block_latency - etc_latency
    block_latency = one_block_latency - etc_latency
    # full_latency = block_latency * num_blocks + etc_latency
    full_latency = (attn_latency + mlp_latency) * num_blocks + etc_latency
    # etc_latency = one_block_latency - attn_latency - mlp_latency

    latency_table['full'] = full_latency
    latency_table['self_attn'] = attn_latency
    latency_table['mlp'] = mlp_latency
    latency_table['block'] = block_latency
    latency_table['etc'] = etc_latency

    print(f"full Latency: {full_latency:.4f}")
    print(f"attn layer Latency: {attn_latency:.4f}")
    print(f"mlp layer Latency: {mlp_latency:.4f}")
    print(f"block  Latency: {block_latency:.4f}")
    print(f'Etc components Latency : {etc_latency:.4f}')
    
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    result_path = os.path.join(args.result_folder, args.result_file)
    
    with open(result_path, 'w') as f:
        json.dump(latency_table, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--iteration', type=int, default=10,
                        help='')
    parser.add_argument('--generation', action='store_true', help='')
    parser.add_argument('--result_folder', type=str, default='',
                        help='')
    parser.add_argument('--result_file', type=str, default='',
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

