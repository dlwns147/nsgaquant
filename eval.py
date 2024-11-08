
from hqq.models.hf.base import AutoHQQHFModel
from utils.owq.utils.modelutils import load_model
from transformers import AutoModelForCausalLM
# from gptqmodel import GPTQModel
# # from gptqmodel.utils import get_backend

from utils.data_utils import get_loader
from utils.eval_utils import eval_metric

model_path = 'meta-llama'
model_name = 'Llama-2-7b-hf'
model_id = f'{model_path}/{model_name}'

# hqq_path = '/SSD/hqq/Llama-2-7b-hf_3bit_64gs_1axis_qscale_false_qzero_false'
# model = AutoHQQHFModel.from_quantized(hqq_path, device_map='auto')
# print(f'hqq_path : {hqq_path}')

awq_path = '/SSD/awq/Llama-2-7b-hf_w4_g128_fake_4bit_awq.pt'
print(f'awq_path : {awq_path}')
model = AutoModelForCausalLM.from_pretrained(awq_path, torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto')
device = model.device
loader = get_loader('wikitext2', train=False, model=model_id)
ppl = eval_metric(model, 'ppl', loader, device=device, seqlen=2048)
print(f'ppl : {ppl}')


gptq_path = '/SSD/gptq'