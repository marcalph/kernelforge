#include <torch/extension.h>


__global__
void rgb_to_grayscale_kernel(
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ input,
    int width, int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int i = row * width + col;
        int inputOffset = i * 3;
        unsigned char r = input[inputOffset + 0];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];
        // ITU-R BT.709 luminance coefficients
        output[i] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    // image: (H, W, 3) uint8 RGB, CUDA
    TORCH_CHECK(image.device().type() == torch::kCUDA);
    TORCH_CHECK(image.dtype() == torch::kByte);
    TORCH_CHECK(image.size(2) == 3, "Expected RGB input (3 channels)");

    const auto height = image.size(0);
    const auto width  = image.size(1);

    auto result = torch::empty(
        {height, width, 1},
        torch::TensorOptions().dtype(torch::kByte).device(image.device())
    );

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width, height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
