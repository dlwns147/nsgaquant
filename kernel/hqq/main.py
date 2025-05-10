import time
from common_code.eval import load_and_eval_ppl

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_path = "meta-llama"
# model_name = "Llama-2-7B-hf"
# model_name = "Llama-2-7b-hf"
model_name = "Llama-2-70b-hf"
# model_name = "Llama-3.2-1B"
model_id  = f'{model_path}/{model_name}'
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

for bits in [3]:
    # nbits=2
    nbits=bits
    group_size=128
    axis=1

    # skip_linears = ['1.self_attn.v_proj', '1.mlp.down_proj', '31.mlp.down_proj']

    #Load model on the CPU
    ######################################################################################
    from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer

    model     = HQQModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, )
    tokenizer = AutoTokenizer.from_pretrained(model_id,       use_auth_token=hf_auth)
    # import code; code.interact('main', local=dict(globals(), **locals()))

    # model = model.to('cuda')
    # for dataset in ['wikitext2', 'c4']:
    #     ppl = load_and_eval_ppl(model, model_name=model_id, dataset=dataset, tokenizer = tokenizer)
    #     print(f"{dataset} PPL: {ppl:.2f}")

    #Quantize the model
    ######################################################################################
    from hqq.core.quantize import *

    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, quant_scale=False, quant_zero=False, axis=axis)

    start = time.time()

    model.quantize_model(quant_config=quant_config)

    end = time.time()

    print(f"{bits} bit Quantization time: {end - start:.2f} seconds")

    import code; code.interact('main', local=dict(globals(), **locals()))

    # import os
    # save_dir = os.path.join('/SSD/Woo/hqq', f'{model_name}_{nbits}bit_{group_size}gs_{axis}axis')
    # from hqq.models.hf.base import AutoHQQHFModel
    # AutoHQQHFModel.save_quantized(model, save_dir)

    # model = model.to('cuda:0')
    
    for dataset in ['wikitext2', 'c4']:
        ppl = load_and_eval_ppl(model, model_name=model_id, dataset=dataset, tokenizer = tokenizer)
        print(f"{dataset} PPL: {ppl:.2f}")

    

    #Evaluate the quantized model 
    ######################################################################################
    # from eval_model import eval_wikitext2
    # eval_wikitext2(model, tokenizer, verbose=True)

