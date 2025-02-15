import torch
from .awq import AWQ
from .gptq import GPTQ
import gc

METHOD = {
    'gptq': GPTQ,
    'awq': AWQ
}

def get_quantized_model(method, arch, model_name, device_map, group_size=128, config=None, dev='cuda', prune=False, do_owq=False, owq_path=None, **kwargs):
    method = METHOD[method](model_name=model_name, config=config, device_map=device_map, group_size=group_size, dev=dev, arch=arch, prune=prune, do_owq=do_owq, owq=owq_path, **kwargs)

    if prune:
        print('Pruning the model')
        method.prune_model()
        
    method.run()
    model = method.model
    del method
    torch.cuda.empty_cache
    gc.collect()

    return model