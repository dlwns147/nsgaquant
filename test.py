
from transformers import AutoModelForCausalLM
from utils.data_utils import get_loader
from utils.eval_utils import eval_metric, get_logits

fp16_model_path = 'meta-llama/Llama-2-7b-hf'
awq_2bit_mode_path = '/SSD/awq/Llama-2-7b-hf_w2_g128_fake_2bit_awq.pt'
orig_model = AutoModelForCausalLM.from_pretrained(fp16_model_path, torch_dtype='auto', device_map='auto')
quant_model = AutoModelForCausalLM.from_pretrained(awq_2bit_mode_path, torch_dtype='auto', device_map='auto')