from torch.utils.cpp_extension import load_inline
from pathlib import Path
import re

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b; }
'''


def get_sig(fname, src):
    res = re.findall(rf'^(.+\s+{fname}\s*\(.*?\))\s*\{{?$', src, re.MULTILINE)
    return res[0]+';' if res else None


def load_cuda_cell(cuda_src, cpp_src, funcs, opt=False, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name=name)


def compile_extension(cuda_source, cpp_source=None, funcs=None, name=None):
    cuda_source = Path(cuda_source)
    if name is None:
        name = cuda_source.stem
    src = cuda_source.read_text()

    if funcs is None:
        funcs = [name]
    if cpp_source is None:
        cpp_source = [get_sig(f, src) for f in funcs]
        cpp_source = [s for s in cpp_source if s is not None]

    cuda_source = cuda_begin + src

    return load_inline(
        name=name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=funcs,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
