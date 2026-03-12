from torch.utils.cpp_extension import load_inline
from pathlib import Path

def compile_extension(cuda_source = "kernelforge/src/kernels/grayscale.cu",
    cpp_source = "torch::Tensor grayscale(torch::Tensor image);"):
    cuda_source = Path(cuda_source)
    name = cuda_source.stem
    cuda_source = cuda_source.read_text()

    return load_inline(
        name=name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=[name],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
