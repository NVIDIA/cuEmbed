# cuEmbed: embedding lookup kernel library

## Overview
cuEmbed is an open-source, header-only CUDA kernel library that accelerates embedding lookup. It aims to achieve high memory bandwidth utilization by maximizing loads in flight when accessing embedding rows. It makes extensive use of C++ templates and compile-time specialization to support a variety of embedding lookup configurations using only a small number of kernels optimized for memory-level parallelism. All of this is intended to make it easy for developers to achieve high performance on embedding lookups in their CUDA programs. 

Supported Operations:
- Forward propagation (fixed-hotness or CSR index formats).
- Backward propagation (COO index format, full or compressed gradients).
- Index transformations (e.g., transpose).

## Development Status

`cuEmbed` is still under development. We aim to keep the host API stable. Users should expect changes in the kernel API and corresponding abstractions of operations.

## How to use
Core components of cuEmbed are the kernel headers in the `cuembed/include` directory. These files have minimal dependency on third-party libraries and are safe to be copied into separate libraries.

### Adding `cuEmbed` to a CMake Project
We recommend using [CMake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake) to fetch cuEmbed into your project. With CPM, getting cuEmbed is easy:
```
CPMAddPackage(
  NAME cuembed
  GIT_REPOSITORY https://rep_ro:${GITLAB_TOKEN}@gitlab-master.nvidia.com/compute/psx/recommender/cuembed.git
  GIT_TAG main
  OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCHMARKS OFF"
)

target_link_libraries(my_library ${cuembed_SOURCE_DIR})
```

### Example usage: Forward Propagation
The following example from `utils/src/embedding_allocation.cu` covers the basic usage of the host API for running forward propagation:
```cpp
template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void RunForward(const utils::AllocationOptions& options,
                const thrust::device_vector<ElemT>& embedding,
                const thrust::device_vector<IndexT>& indices,
                const thrust::device_vector<OffsetT>& offsets,
                const thrust::device_vector<ElemT>& weights,
                thrust::device_vector<ElemT>* result) {
  const int* offsets_ptr = nullptr;
  int hotness = options.hotness();
  if (options.is_csr()) {
    offsets_ptr = offsets.data().get();
    hotness = 0;
  }
  const ElemT* weight_ptr = nullptr;
  if (options.is_weighted()) {
    weight_ptr = weights.data().get();
  }
  using InputT = ElemT;
  using OutputT = ElemT;
  EmbeddingForward<InputT, OutputT, IndexT, OffsetT, fp16_math>(
      embedding.data().get(),
      options.embed_width(),
      indices.data().get(),
      offsets_ptr,
      weight_ptr,
      options.batch_size(),
      hotness,
      options.combine_mode(),
      result->data().get());
}
```
In the above example, we call `EmbeddingForward` with the corresponding data pointers from the embedding table (i.e., `embedding`), the embedding row indices (i.e., `indices`) & offsets indicating the starting position of each set of indices (i.e., `offsets`) & per sample weights (i.e., `weights`), the output of embedding lookup (i.e., `result`), and workload descriptions (i.e., `embedding_width`, `hotness`, `batch_size`, `combine_mode` unwrapped from `options`). The end result of embedding lookup is written into `result`. 

Please refer to `utils/src/embedding_allocation.cu` for more examples, including index transposition and backward propagation. 

Detailed descriptions of the full API and parameters can be found in [cuembed/README.md](https://gitlab-master.nvidia.com/compute/psx/recommender/cuembed/-/blob/main/cuembed/README.md?ref_type=heads).

## Building cuEmbed tests and benchmarks
Since cuEmbed is header-only, there is nothing to build to use it.
To build the tests and benchmarks: 

### Build From Source
```bash
git clone --recursive https://gitlab-master.nvidia.com/compute/psx/recommender/cuembed
cd cuembed
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
Binaries will be built into:
- `build/tests`
- `build/benchmarks`

## Benchmarks
### Full Suite Benchmarks
To run benchmarks locally:
```bash
cd benchmarks/
./sweep_parameters.sh
```

### Manual Benchmark Single Test Case

Manual benchmarking can be done with the `manual_benchmark` binary in the `benchmarks` folder. This will run the forward, transpose, and backward stages. 

Example: 
```bash
./bin/benchmarks/manual_benchmark --num_categories 10000000 --embed_width 256 --batch_size 65536 --alpha=1.15 --hotness=64 --csr_input=false --half_embedding_type=true --weighted_sum=false --compressed_grad=true
```

## Detailed Support Matrix
|                             | Supported In Current Release         | Future Release                       |
|-----------------------------|-------------------------|--------------------------------------|
| Embedding table size        | single table single GPU | multiple tables and multiple devices |
| Embedding cache integration | no                      | yes                                  |
| Embedding & Output types    | fp32, fp16              | bf16                                 |
| Lookup Index types          | int32_t, int64_t        |                                      |
| Lookup Index Layout (fwd)   | fixed hotness, CSR      | COO                                  |
| Lookup Index Layout (bwd)   | COO                     |                                      |
| Reduction type (fwd)        | weighted sum, concat, mean             |                       |
| Reduction type (bwd)        | weighted sum, concat    | mean         |
| Reduction precision         | fp32, fp16              | bf16                                 |
| Kernel type                 | fwd, bwd, transpose     | optimizer                            |

## Requirements
- nvcc 12.0+
- C++ 17
- Volta+
