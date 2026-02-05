# KernelForge: CUDA Development on Google Colab via VS Code

Goal: Develop and profile CUDA kernels on Google Colab from VS Code locally

## Phase 1: VS Code Setup

### Install Extensions
- [ ] Open VS Code Extensions (`Cmd+Shift+X`)
- [ ] Search for "Google Colab" and install the official extension (`Google.colab`)
- [ ] Install Jupyter extension if prompted (dependency)

### Connect to Colab Runtime
- [ ] Open a `.ipynb` notebook in VS Code
- [ ] Click "Select Kernel" in the top right
- [ ] Choose "Connect to Google Colab"
- [ ] Sign in with your Google account
- [ ] Select runtime type:
  - Free tier: T4 GPU
  - Pro tier: A100, V100, etc.

## Phase 2: Verify GPU Environment

Run in a notebook cell:
```python
!nvidia-smi
!nvcc --version
!nvidia-smi --query-gpu=name,compute_cap --format=csv
```

GPU compute capabilities (for nvcc `-arch` flag):
- T4: sm_75
- V100: sm_70
- A100: sm_80
- L4: sm_89

## Phase 3: Project Setup

### Option A: Write kernels in notebook cells
```python
%%writefile vector_add.cu
#include <stdio.h>

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}

int main() {
    // ... kernel code
}
```

### Option B: Clone repo to Colab filesystem
```python
!git clone https://github.com/YOUR_USERNAME/kernelforge.git
%cd kernelforge
```

### Option C: Mount Google Drive for persistence
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/kernelforge
```

## Phase 4: Compile and Run

### Basic Compilation
```bash
!nvcc -o vector_add vector_add.cu
!./vector_add
```

### With Architecture Flag (recommended)
```bash
# For T4 (most common free tier)
!nvcc -arch=sm_75 -o vector_add vector_add.cu

# For A100
!nvcc -arch=sm_80 -o vector_add vector_add.cu
```

### Debug Build
```bash
!nvcc -g -G -o vector_add_debug vector_add.cu
```

## Phase 4: Profiling

### Basic Timing with nvprof (deprecated but available)
```bash
!nvprof ./vector_add
```

### Nsight Systems (recommended for timeline)
```bash
# Install if not available
!apt-get install -y nsight-systems

# Profile
!nsys profile --stats=true ./vector_add

# Output will be in report*.nsys-rep
```

### Nsight Compute (recommended for kernel analysis)
```bash
# Basic kernel profiling
!ncu ./vector_add

# Detailed metrics
!ncu --set full ./vector_add

# Save report
!ncu -o profile_report ./vector_add
```

### CUDA Events (in-code timing)
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %f ms\n", milliseconds);
```

## Phase 5: Iterate & Learn

- [ ] Add more complex kernels:
  - [ ] Matrix multiplication
  - [ ] Reduction
  - [ ] Shared memory usage
  - [ ] Warp-level primitives (`__shfl_*`)
- [ ] Profile and optimize:
  - [ ] Memory coalescing
  - [ ] Occupancy tuning
  - [ ] Bank conflict analysis
- [ ] Compare with CPU baseline

## Quick Reference

### Colab-Specific Commands
```python
# Write CUDA file from notebook cell
%%writefile kernel.cu

# Run shell command
!nvcc -o kernel kernel.cu

# Time a cell
%%time
!./kernel

# Suppress output
%%capture
!nvcc -o kernel kernel.cu
```

### Useful nvcc Flags
```bash
-arch=sm_XX       # Target GPU architecture
-g -G             # Debug symbols (host + device)
-lineinfo         # Line info for profiler
-O3               # Optimization level
--ptxas-options=-v # Show register/memory usage
-Xcompiler -fopenmp # OpenMP on host
```

### Memory Debugging
```bash
# Check for memory errors
!compute-sanitizer ./vector_add

# Detailed race detection
!compute-sanitizer --tool racecheck ./vector_add

# Memory leak check
!compute-sanitizer --tool memcheck ./vector_add
```

### Colab Limitations
- Free tier: ~12 hour session limit, GPU may disconnect
- Free GPU: Usually T4 (16GB), sometimes K80 (older)
- No persistent storage: save work to Drive or GitHub
- Background execution not supported: stay on the tab
