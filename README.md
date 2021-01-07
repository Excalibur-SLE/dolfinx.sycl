# Dolfinx.SYCL
Simple code to assemble the Poisson equation on manycore architectures using Dolfinx and SYCL.

## Requirements:
  - FEniCS/DOLFIN-X installation (development version of dolfinx required)
  - A SYCL implementation

### Supported SYCL Implementations
Supported (tested) SYCL implementation:
- hipSYCL
- LLVM 
- LLVM-CUDA


## Building

### Using the hipSYCL implementation
Building for CPUs:
```bash
export HIPSYCL_PLATFORM=omp

mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

Building with CUDA and Nvidia Tesla P100 GPU accelerator:
```bash
export HIPSYCL_PLATFORM=cuda
export HIPSYCL_GPU_ARCH=sm_60

mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DCUDA_PATH=${CUDA_PATH} ..
make -j8
```
** Runnning Dolfinx.sycl with hipsycl + CUDA requires eigen@master.

### Using Intel SYCL implementation
```bash
export CXX=clang++
export CC=gcc

mkdir build
cd build
cmake -DSYCL_IMPL=LLVM ..
make -j8
```

Using Intel SYCL with CUDA:
```bash

export CXX=clang++
export CC=gcc

mkdir build
cd build
cmake -DSYCL_IMPL=LLVM-CUDA -DCUDA_PATH=${CUDA_PATH} ..
make -j8
```


## Runinng
```bash
./dolfinx_sycl {Ncells} {platform}
```
{platform} - Platform to run on [cpu or gpu].

{Ncells} - Number of cells in each direction ($`N_x`$, $`N_y`$, $`N_z`$), default is 50. 
Totalizing 750000 cells ($`6 \times N_x \times N_y \times N_z`$).

## Limitations
Too many to mention ...
- Assemble on cells only

## Docker
```bash
docker run --gpus all -v $(pwd):/home/fenics/shared --name sycl igorbaratta/dolfinx_sycl:latest nvidia-smi
```

## Singularity

```bash
singularity pull library://igorbaratta/default/dolfinx_sycl:latest
singularity run --nv dolfinx_sycl_latest.sif
```