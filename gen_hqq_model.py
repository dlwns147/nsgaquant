#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_path = "/SSD/huggingface/meta-llama"
# model_name = "Llama-2-7b-hf"
model_name = "Llama-2-13b-hf"
model_id  = f'{model_path}/{model_name}'
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

nbits=2
group_size=64
axis=1


quant_scale = False
quant_zero = False

saved_model_path = f"/SSD/hqq/Llama-2-7b-hf_{nbits}bit_{group_size}gs_{axis}axis_qscale_false_qzero_false"

#Load model on the CPU
######################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer

model     = HQQModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id,       use_auth_token=hf_auth) 

#Quantize the model
######################################################################################
from hqq.core.quantize import *

quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis, quant_scale=quant_scale, quant_zero=quant_zero)
model.quantize_model(quant_config=quant_config)

# model = AutoHQQHFModel.from_quantized(saved_model_path).to(device)
import os
save_dir = os.path.join('/SSD/hqq', f'{model_name}_{nbits}bit_{group_size}gs_{axis}axis_qscale_{str(quant_scale).lower()}_qzero_{str(quant_zero).lower()}')
from hqq.models.hf.base import AutoHQQHFModel
AutoHQQHFModel.save_quantized(model, save_dir)

#Evaluate the quantized model 
######################################################################################

# from eval_utils import load_and_eval_ppl
# ppl = load_and_eval_ppl(model=model, dataset='wikitext2', tokenizer=tokenizer)
# print(f'ppl : {ppl}')
# from eval_model import eval_wikitext2
# eval_wikitext2(model, tokenizer, verbose=True) 