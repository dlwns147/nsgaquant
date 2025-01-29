#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
// #include <torch/extension.h>
#include <torch/library.h>

#define WARP_SIZE 32

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT2 =  16;
const int BLOCKHEIGHT3 =  24;
const int BLOCKHEIGHT4 =  32;
const int BLOCKHEIGHT8 =  64;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__global__ void VecQuant3MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();
  if (w < width) {
    while (k < blockwidth2) {
        int g = (g_h + (k * 2)) / groupsize;
        float scale_f = scales[g * width + w];
        float zero_f = zeros[g * width + w];
        half2 scale = __float2half2_rn(scale_f);
        half2 zero = __float2half2_rn(-(zero_f));

        std::memset(&res2, 0, sizeof(half2));
        tmp1 = as_unsigned(mat[i]);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
        i += width;
        tmp2 = as_unsigned(mat[i]);
        tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
        res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
        tmp2 >>= 4;
        k += 6;
        res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
        i += width;
        tmp1 = as_unsigned(mat[i]);
        tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
        res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
        tmp1 >>= 2;
        k += 5;
        res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
        res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
        i += width;
        k += 5;
        res += __low2float(res2) + __high2float(res2);
    }

    atomicAdd(&mul[b * width + w], res);
  }
}

// __global__ void VecQuant3OutlierMatMulKernel(
//     const  half2* __restrict__ vec,
//     const    int* __restrict__ mat,
//            float* __restrict__ mul,
//     const  float* __restrict__ scales,
//     const  float* __restrict__ zeros,
//     const   half* __restrict__ outlierMat,
//     const    int* __restrict__ outlieridx,
//     const    int* __restrict__ outrow,
//     const    int* __restrict__ cnt,
// 	int batch,
// 	int vec_height,
//     int height,
//     int width,
//     int groupsize
// ) {
//   const int blockwidth2 = BLOCKWIDTH / 2;
//   int b = blockIdx.z;
//   int h = BLOCKHEIGHT3 * blockIdx.x;
//   int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

//   __shared__ half2 blockvec[blockwidth2];
//   if (threadIdx.x < blockwidth2)
//     blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

//   __shared__ half2 deq2[64][32];
//   int val = threadIdx.x / 32;
//   int off = threadIdx.x % 32;
//   for (; val < 64; val += BLOCKWIDTH / 32) {
//     deq2[val][off] = __halves2half2(
//        __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
//     );
//   }

//   // -----for outlier weight-----
//   int flag = 0, oidx = 0;
//   int blockoutrow = outrow[blockIdx.x];
//   int blockcnt = cnt[blockIdx.x];

//   outlierMat += blockoutrow * width;
//   outlieridx += blockoutrow;

//   for (int i = 0, outidx = 0; i < blockcnt; i++){
//     outidx = outlieridx[i];
//     if (threadIdx.x == (outidx / 2) % blockwidth2){
//       flag += (outidx % 2) + 1; // 1 (x) 2 (y) 3(x,y)
//       oidx = i;
//     }
//   }
  
//   __shared__ half2 blockvec[blockwidth2];
//   __shared__ half blockveco[MAXOUTLIER];

//   if (threadIdx.x < bwidth){
//     blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];
//     if (flag == 1)
//       blockveco[oidx] = blockvec[threadIdx.x].x;
//     else if (flag == 2)
//       blockveco[oidx] = blockvec[threadIdx.x].y;
//     else if (flag == 3){
//       blockveco[oidx - 1] = blockvec[threadIdx.x].x;
//       blockveco[oidx] = blockvec[threadIdx.x].y;
//     }
//   }
//   // ---------------------------
  
//   int i = width * h + w;
//   int g_h = (h / 3) * 32;
//   int k = 0;

//   float res = 0;
//   half2 res2;

//   unsigned int tmp1;
//   unsigned int tmp2;
//   unsigned int tmp;

//   __syncthreads();

//   while (k < blockwidth2) {
//     int g = (g_h + (k * 2)) / groupsize;
// 	float scale_f = scales[g * width + w];
// 	float zero_f = zeros[g * width + w];
//     half2 scale = __float2half2_rn(scale_f);
//     half2 zero = __float2half2_rn(-(zero_f));

//     std::memset(&res2, 0, sizeof(half2));
//     tmp1 = as_unsigned(mat[i]);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
//     i += width;
//     tmp2 = as_unsigned(mat[i]);
//     tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
//     res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
//     tmp2 >>= 4;
//     k += 6;
//     res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
//     i += width;
//     tmp1 = as_unsigned(mat[i]);
//     tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
//     res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
//     tmp1 >>= 2;
//     k += 5;
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
//     res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
//     i += width;
//     k += 5;
//     res += __low2float(res2) + __high2float(res2);
//   }

//   atomicAdd(&mul[b * width + w], res);
// }

torch::Tensor vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(vec.device()); 
  torch::Tensor mul = torch::zeros({batch, width}, options);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernel<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<float>(),
    batch, vec_height, height, width, groupsize
  );

  return mul;
}

torch::Tensor vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat,
  torch::Tensor scales, torch::Tensor zeros,
  int64_t groupsize, int64_t vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  return vecquant3matmul_cuda(vec, mat, scales, zeros, groupsize, vec_height);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
// }

TORCH_LIBRARY(custom_gptq, m) {
  m.def("vecquant3matmul(Tensor vec, Tensor mat, Tensor scales, Tensor zeros, int groupsize, int vec_height) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_gptq, CUDA, m) {
  m.impl("vecquant3matmul", &vecquant3matmul);
}