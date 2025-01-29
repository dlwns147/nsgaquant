import torch
from torch import Tensor

# from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

# lib = torch.library.Library("custom_gptq", "FRAGMENT")
# lib.define("vecquant3matmul(Tensor vec, Tensor mat, Tensor scales, Tensor zeros, int groupsize, int vec_height) -> Tensor")

# torch.ops.load_library('/NAS/JG/QAS4SD/GPTQ_kernels/build/lib.linux-x86_64-cpython-311/custom_gptq.cpython-311-x86_64-linux-gnu.so')
"""
우승택 수정 torch.ops
"""
# torch.ops.load_library('/NAS/JG/QAS4SD/GPTQ_kernels/build/lib.linux-x86_64-cpython-311/custom_gptq/_C.cpython-311-x86_64-linux-gnu.so')
# torch.ops.load_library('/NAS/JG/QAS4SD/GPTQ_kernels/build/lib.linux-x86_64-cpython-310/custom_gptq/_C.cpython-310-x86_64-linux-gnu.so')
torch.ops.load_library('/NAS/SJ/nsgaquant/kernel/GPTQ_kernels/build/lib.linux-x86_64-cpython-310/custom_gptq/_C.cpython-310-x86_64-linux-gnu.so')

# @torch.library.custom_op("custom_gptq::vecquant3matmul", mutates_args=())
# @torch.jit.ignore
def vecquant3matmul(vec: Tensor, mat: Tensor, scales: Tensor, zeros: Tensor, groupsize: int, vec_height: int) -> Tensor:
    # import code; code.interact('vecquant3matmul', local=dict(globals(), **locals()))
    return torch.ops.custom_gptq.vecquant3matmul.default(
        vec, mat, scales, zeros, groupsize, vec_height,
    )
    # return custom_gptq.vecquant3matmul(vec,mat,scales,zeros,groupsize,vec_height)
    
@torch.library.register_fake("custom_gptq::vecquant3matmul")
def _(vec: Tensor, mat: Tensor, scales: Tensor, zeros: Tensor, groupsize: int, vec_height: int) -> Tensor:
    BS, IC = vec.shape
    _, OC = mat.shape
    return vec.new_empty((BS, OC))


