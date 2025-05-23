# Source: llm-awq github (https://github.com/mit-han-lab/llm-awq)

import torch
import accelerate
import gc


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module
    for name, buffer in model.named_buffers():
        if name.endswith(module_name):
            return buffer


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        # print(f'n: {n}, d: {d}, m: {m}')
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            # if m is a block, replace hqq meta data
            for n, submodule in m.named_modules():
                if hasattr(submodule, 'meta'):
                    # print(f'{n} : {submodule}')
                    submodule.meta['scale'] = submodule.meta['scale'].to(d)
                    submodule.meta['zero'] = submodule.meta['zero'].to(d)
                if hasattr(submodule, 'bias'):
                    if isinstance(submodule.bias, torch.nn.Parameter):
                        submodule.bias.data = submodule.bias.data.to(d)
                    if isinstance(submodule.bias, torch.Tensor):
                        submodule.bias = submodule.bias.to(d)
                    # print(f"d : {d}, meta['scale'] : {submodule.meta['scale'].device}, meta['zero'] : {submodule.meta['zero'].device}")
            add_hook_to_module(m, hook)
    gc.collect()
    torch.cuda.empty_cache()
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model
