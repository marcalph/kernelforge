#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}



__global__
void blur_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int linOffset = height*width*channels;

    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-radius; blurRow<=radius; blurRow++){
            for (int blurCol=-radius; blurCol<=radius; blurCol++){
                int curRow = row+blurRow;
                int curCol = col+blurCol;
                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    pixVal += input[linOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }

        output[linOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}



torch::Tensor blur(torch::Tensor image, int radius) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius>0);

    
    CHECK_INPUT(image)
    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    blur_kernel<<<number_of_blocks, threads_per_block>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height, 
        radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}