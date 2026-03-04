#include <torch/extension.h>


// Each thread loads one RGBA pixel as a single uchar4 (32-bit / 4-byte transaction)
// instead of 3 separate byte loads for RGB — better memory coalescing.
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
        uchar4 px = reinterpret_cast<const uchar4*>(input)[i];  // 128-bit load (4 bytes)
        // ITU-R BT.709 luminance coefficients
        output[i] = (unsigned char)(0.2126f * px.x + 0.7152f * px.y + 0.0722f * px.z);
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    // image: (H, W, 4) uint8 RGBA, CUDA
    TORCH_CHECK(image.device().type() == torch::kCUDA);
    TORCH_CHECK(image.dtype() == torch::kByte);
    TORCH_CHECK(image.size(2) == 4, "Expected RGBA input (4 channels)");

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
