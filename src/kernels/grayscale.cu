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
        // ITU-R BT.709 luminance
        output[i] = (unsigned char)(0.2126f * input[inputOffset]
                                  + 0.7152f * input[inputOffset + 1]
                                  + 0.0722f * input[inputOffset + 2]);
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    TORCH_CHECK(image.device().type() == torch::kCUDA);
    TORCH_CHECK(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width  = image.size(1);

    auto result = torch::empty({height, width, 1},
        torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0,
                              c10::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width, height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
