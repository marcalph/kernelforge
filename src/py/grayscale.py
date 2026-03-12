
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from torchvision.io import read_image, write_png


def show_img(x, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1, 2, 0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)





def rgb2gray_py(x):
    c,h,w = x.shape
    n = h*w
    x = x.flatten()
    res = torch.empty(n, dtype=x.dtype, device=x.device)
    for i in range(n): res[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]
    return res.view(h,w)
     

def main(ext):
    """
    Read input image, convert it to grayscale via custom CUDA kernel and write out as png.
    """

    x = read_image("kernelforge/pisco.jpg").permute(1, 2, 0).cuda()
    print("Input image:", x.shape, x.dtype, "mean:", x.float().mean().item())

    assert x.dtype == torch.uint8

    y = ext.rgb_to_grayscale(x)

    print("Output image:", y.shape, y.dtype, "mean:", y.float().mean().item())
    return y.cpu()