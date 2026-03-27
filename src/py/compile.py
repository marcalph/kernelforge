from torch.utils.cpp_extension import load_inline
from pathlib import Path


cuda_begin = r"""
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) {return (a+b-1)/b;}
"""


def compile_extension(cuda_source ,
    cpp_source , 
    funcs, name = None):
    cuda_source = Path(cuda_source)
    if name is None:
        name  = cuda_source.stem
    cuda_source = cuda_source.read_text()
    cuda_source = cuda_begin + cuda_source

    return load_inline(
        name=name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=[name],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
