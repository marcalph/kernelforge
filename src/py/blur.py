

import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from torchvision.io import read_image, write_png




def main(ext):
    """
    Read input image, convert it to grayscale via custom CUDA kernel and write out as png.
    """

    x = read_image("kernelforge/pisco.jpg").permute(1, 2, 0).cuda()
    print("Input image:", x.shape, x.dtype, "mean:", x.float().mean().item())

    assert x.dtype == torch.uint8

    y = ext.blur(x, 8)

    print("Output image:", y.shape, y.dtype, "mean:", y.float().mean().item())
    return y.cpu()