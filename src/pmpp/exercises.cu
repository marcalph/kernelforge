void chap2_1() {
    // mapping thread/block to data index i
    i = blockIdx.x * blockDim.x + threadIdx.x;
}

void chap2_2(){
    // mapping thread/block to first data index if we want to process elts 2 by 2
    i = (blockIdx.x * blockDim.x +threadIdx.x) * 2;
}

void chap2_3 () {
    i=blockIdx.x * blockDim.x * 2 + threadIdx.x;
}