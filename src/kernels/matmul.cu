#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__
void matmul_k(float *m, float *n, float * out, int h, int w, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r>=h || c>= w) return; // out of bound
    float dotval=0;
    for (int i=0;i<k;i++){
        dotval+= m[r*k+i] * n[i*k+c];
    }
    out[r*w+c]=dotval;
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);

    int i = m.size(0); 
    int k = m.size(1);
    int j = n.size(1);
    
    // Ensure matrices are compatible for multiplication
    TORCH_CHECK(k == n.size(0), "Size mismatch!");

    // Initialize output tensor
    auto output = torch::zeros({i, j}, m.options());

    // Define thread block and grid dimensions
    dim3 tpb(16, 16);
    dim3 blocks(cdiv(j, tpb.x), cdiv(i, tpb.y));

    // Launch CUDA kernel
    matmul_k<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), i, j, k);

        // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}