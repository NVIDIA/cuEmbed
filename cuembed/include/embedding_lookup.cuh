// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format on

//! \file
#ifndef CUEMBED_INCLUDE_EMBEDDING_LOOKUP_CUH_
#define CUEMBED_INCLUDE_EMBEDDING_LOOKUP_CUH_

#include <cstdlib>
#include <iostream>
#include <tuple>

#include "cuembed/include/embedding_lookup_kernels.cuh"
#include "cuembed/include/embedding_lookup_types.cuh"

namespace cuembed {

#define MAX_THREADS_PER_CTA 1024
// With 48 registers per CTA, we can run at an occupancy of 40 warps/SM.
// Thus we want to have at most 8 warps per CTA to achieve best occupancy.
#define DEFAULT_THREADS_PER_CTA 256

// Shmem for a single CTA is at most 48KB.
#define MAX_SHMEM_BYTES_PER_CTA (48 * 1024)

#define EMBEDDING_LOOKUP_DISPATCH(mode, element_per_load)                      \
  using InputVecT =                                                            \
      typename VecTypeHelper<InputT, element_per_load, fp16_math>::LoadType;   \
  using ReduceVecT =                                                           \
      typename VecTypeHelper<InputT, element_per_load, fp16_math>::ReduceType; \
  using OutputVecT =                                                           \
      typename VecTypeHelper<OutputT, element_per_load, fp16_math>::LoadType;  \
  using AddresserT =                                                           \
      Addresser<InputT, OutputT, IndexT, InputVecT, OutputVecT, mode>;         \
  using CombinerT = Combiner<InputVecT, ReduceVecT, OutputVecT, mode>;         \
  if (offsets == nullptr && weights != nullptr && num_hots >= 8) {             \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, void>;                     \
    constexpr int UnrollFactor = 8;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           nullptr,            \
                                                           num_hots,           \
                                                           weights,            \
                                                           ret);               \
  } else if (offsets != nullptr && weights != nullptr && num_hots >= 8) {      \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, OffsetT>;                  \
    constexpr int UnrollFactor = 8;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           offsets,            \
                                                           num_hots,           \
                                                           weights,            \
                                                           ret);               \
  } else if (offsets == nullptr && weights == nullptr && num_hots >= 8) {      \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, void>;                     \
    constexpr int UnrollFactor = 8;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           nullptr,            \
                                                           num_hots,           \
                                                           nullptr,            \
                                                           ret);               \
  } else if (offsets != nullptr && weights == nullptr && num_hots >= 8) {      \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, OffsetT>;                  \
    constexpr int UnrollFactor = 8;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           offsets,            \
                                                           num_hots,           \
                                                           nullptr,            \
                                                           ret);               \
  } else if (offsets == nullptr && weights != nullptr && num_hots < 8) {       \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, void>;                     \
    constexpr int UnrollFactor = 4;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           nullptr,            \
                                                           num_hots,           \
                                                           weights,            \
                                                           ret);               \
  } else if (offsets != nullptr && weights != nullptr && num_hots < 8) {       \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, OffsetT>;                  \
    constexpr int UnrollFactor = 4;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           offsets,            \
                                                           num_hots,           \
                                                           weights,            \
                                                           ret);               \
  } else if (offsets == nullptr && weights == nullptr && num_hots < 8) {       \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, void>;                     \
    constexpr int UnrollFactor = 4;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           nullptr,            \
                                                           num_hots,           \
                                                           nullptr,            \
                                                           ret);               \
  } else if (offsets != nullptr && weights == nullptr && num_hots < 8) {       \
    using IndexLoaderT = IndexLoader<IndexT, ElemT, OffsetT>;                  \
    constexpr int UnrollFactor = 4;                                            \
    EmbeddingLookUpKernel<IndexLoaderT, AddresserT, CombinerT, UnrollFactor>   \
        <<<launch_grid, launch_block, smem_size, stream>>>(params,             \
                                                           embed_width,        \
                                                           batch_size,         \
                                                           indices,            \
                                                           offsets,            \
                                                           num_hots,           \
                                                           nullptr,            \
                                                           ret);               \
  } else {                                                                     \
    CUEMBED_ASSERT(false);                                                     \
  }

#define CUEMBED_ASSERT(condition)                                           \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::cerr << "Check failed: " #condition << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                   \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

template <typename ElemT>
std::tuple<int, int> DivideRowIntoVectors(const int embed_width) {
  const size_t bytes_per_row = embed_width * sizeof(ElemT);
  CUEMBED_ASSERT(bytes_per_row % 4 == 0);

  // Targeting LDG.E.128 for each load instruction.
  // Thus requiring each load to be 16 bytes.
  // If not possible, reduce the load width.
  int bytes_per_load = 8;
  if ((bytes_per_row % 16 == 0) && (bytes_per_row > 64)) {
    bytes_per_load = 16;
  } else if (bytes_per_row % 8 == 0) {
    bytes_per_load = 8;
  } else if (bytes_per_row % 4 == 0) {
    bytes_per_load = 4;
  }

  const int element_per_load = bytes_per_load / sizeof(ElemT);
  const int threads_per_row = embed_width / element_per_load;
  CUEMBED_ASSERT(threads_per_row <= MAX_THREADS_PER_CTA);
  return std::make_tuple(element_per_load, threads_per_row);
}

/*!
 * \brief  Heuristics for launch parameters are wrapped in this function.
 */
template <typename ElemT, typename IndexT>
std::tuple<int, int, int> GetKernelLaunchParams(const int embed_width,
                                                const int num_hots,
                                                const bool is_weighted) {
  auto [element_per_load, threads_per_sample] =
      DivideRowIntoVectors<ElemT>(embed_width);

  // TODO(zejiaz): with super wide embeddings we should have another kernel that
  // unrolls in embedding row dimension.
  int samples_per_cta = 1;
  if (threads_per_sample <= DEFAULT_THREADS_PER_CTA) {
    samples_per_cta = DEFAULT_THREADS_PER_CTA / threads_per_sample;
  }

  auto weight_size = is_weighted ? sizeof(ElemT) : 0;
  while ((samples_per_cta * num_hots * (sizeof(IndexT) + weight_size)) >=
         MAX_SHMEM_BYTES_PER_CTA) {
    samples_per_cta /= 2;
  }
  CUEMBED_ASSERT(samples_per_cta > 0);

  return std::make_tuple(element_per_load, threads_per_sample, samples_per_cta);
}

/**
 * @brief Embedding forward propagation function. Handles Fixed and CSR formats.
 * Current assumptions:
 *    Shared memory reserved is large enough to store the CTA's lookup indices.
 *    Embedding row width is <= 16K Bytes.
 *    Embedding row width is divisible by 4 Bytes.
 *    If not null, weights array is of the same size as indices array.
 *    Fixed embedding row width for all embedding vectors in the single
 * embedding table.
 *
 * @tparam InputT Params tensor datatype
 * @tparam OutputT Result tensor datatype
 * @tparam IndexT Index datatype
 * @tparam OffsetT CSR offset datatype
 * @tparam fp16_math Flag to perform reduction in fp16
 *
 * @param params Pointer to the embedding table data.
 * @param embed_width Number of elements in each embedding row.
 * @param indices Pointer to the lookup indices.
 * @param offsets Pointer to the offsets (CSR format). Must be nullptr when
 * launching for fixed hotness.
 * @param weights Pointer to the weight array. Weight for a specific lookup
 * index is applied to the loaded embedding row before reduction. If nullptr,
 * will use just the embedding row for reduction. The type for the weights must
 * the be the same as the input type. If the input type is structured, then the
 * user need to define their own GetElemT<InputT> specialization.
 * @param batch_size  Batch size of the embedding lookup workload.
 * @param num_hots Number of rows to lookup for each sample in batch. Must be 0
 * when launching for CSR indices layout.
 * @param mode ReductionType::kSum (computes the summation of the looked up rows
 * for each sample) or ReductionType::kConcat (concatenates all looked up rows).
 * @param ret Pointer to the output location.
 * @param stream Optional. The cudaStream to launch the kernel asynchronously.
 * If not specified, will launch the kernel on default stream.
 */
template <typename InputT,
          typename OutputT,
          typename IndexT,
          typename OffsetT,
          bool fp16_math = false>
void EmbeddingForward(const InputT* params,
                      const int embed_width,
                      const IndexT* indices,
                      const OffsetT* offsets,
                      const GetElemT<InputT>* weights,
                      const int batch_size,
                      const int num_hots,
                      const CombineMode mode,
                      OutputT* ret,
                      const cudaStream_t stream = 0) {
  // Concat does not have weighed option.
  CUEMBED_ASSERT(weights == nullptr || mode != CombineMode::kConcat);

  // CSR or fixed hotness.
  CUEMBED_ASSERT((offsets != nullptr && num_hots == 0) ||
                 (offsets == nullptr && num_hots > 0));
  // CSR does not support concat.
  CUEMBED_ASSERT(offsets == nullptr || mode != CombineMode::kConcat);

  using ElemT = GetElemT<InputT>;

  auto [element_per_load, threads_per_sample, samples_per_cta] =
      GetKernelLaunchParams<ElemT, IndexT>(
          embed_width, num_hots, weights == nullptr);
  dim3 launch_block(embed_width / element_per_load, samples_per_cta, 1);
  dim3 launch_grid((batch_size + samples_per_cta - 1) / samples_per_cta, 1, 1);
  size_t smem_size = samples_per_cta * num_hots * sizeof(IndexT);
  if (weights != nullptr) {
    smem_size += samples_per_cta * num_hots * sizeof(ElemT);
  }

  if (mode == CombineMode::kSum && element_per_load == 8) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kSum, 8);
  } else if (mode == CombineMode::kSum && element_per_load == 4) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kSum, 4);
  } else if (mode == CombineMode::kSum && element_per_load == 2) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kSum, 2);
  } else if (mode == CombineMode::kSum && element_per_load == 1) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kSum, 1);
  } else if (mode == CombineMode::kConcat && element_per_load == 8) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kConcat, 8);
  } else if (mode == CombineMode::kConcat && element_per_load == 4) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kConcat, 4);
  } else if (mode == CombineMode::kConcat && element_per_load == 2) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kConcat, 2);
  } else if (mode == CombineMode::kConcat && element_per_load == 1) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kConcat, 1);
  } else if (mode == CombineMode::kMean && element_per_load == 8) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kMean, 8);
  } else if (mode == CombineMode::kMean && element_per_load == 4) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kMean, 4);
  } else if (mode == CombineMode::kMean && element_per_load == 2) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kMean, 2);
  } else if (mode == CombineMode::kMean && element_per_load == 1) {
    EMBEDDING_LOOKUP_DISPATCH(CombineMode::kMean, 1);
  } else {
    CUEMBED_ASSERT(0);
  }
}

#define MAX_NZ_PER_BLOCK 128
#define MIN_NZ_PER_BLOCK 8
#define GRAD_KERNEL_TARGET_THREAD_USAGE 0.4

#define GRAD_EMBEDDING_LOOKUP_DISPATCH(element_per_load)                     \
  using GradVecT =                                                           \
      typename VecTypeHelper<GradT, element_per_load, false>::LoadType;      \
  using GradIndexLoaderT = GradIndexLoader<IndexT, WeightT>;                 \
  using GradAddresserT = GradAddresser<GradT, GradVecT>;                     \
  using GradCombinerT = GradCombiner<GradT, GradVecT, WeightT>;              \
  if (transpose_weights == nullptr) {                                        \
    EmbeddingBackwardKernel<GradIndexLoaderT, GradAddresserT, GradCombinerT> \
        <<<launch_grid, launch_block, smem_size, stream>>>(                  \
            grad_y,                                                          \
            embed_width,                                                     \
            transpose_indices_,                                              \
            transpose_sample_ids,                                            \
            nullptr,                                                         \
            nnz,                                                             \
            nz_block_size,                                                   \
            grad_embedding);                                                 \
  } else {                                                                   \
    EmbeddingBackwardKernel<GradIndexLoaderT, GradAddresserT, GradCombinerT> \
        <<<launch_grid, launch_block, smem_size, stream>>>(                  \
            grad_y,                                                          \
            embed_width,                                                     \
            transpose_indices_,                                              \
            transpose_sample_ids,                                            \
            transpose_weights,                                               \
            nnz,                                                             \
            nz_block_size,                                                   \
            grad_embedding);                                                 \
  }

/*!
 * \brief  Heuristics for backward launch parameters are wrapped in this
 * function.
 */
template <typename GradT, typename IndexT, typename WeightT>
std::tuple<int, int, int, int> GetGradKernelLaunchParams(const int embed_width,
                                                         const int nnz) {
  auto [element_per_load, threads_per_nonzero] =
      DivideRowIntoVectors<GradT>(embed_width);

  // Heuristic to determine nz_block_size
  int device;
  cudaGetDevice(&device);

  int max_threads_per_mp;
  cudaDeviceGetAttribute(
      &max_threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, device);

  int mp_count;
  cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device);

  int64_t max_threads = max_threads_per_mp * mp_count;
  int64_t threshold = max_threads * GRAD_KERNEL_TARGET_THREAD_USAGE;
  int nz_block_size = MAX_NZ_PER_BLOCK;

  // Reduce nz_block_size until num threads greater than threshold
  while ((nnz / nz_block_size) * threads_per_nonzero < threshold) {
    nz_block_size /= 2;
    if (nz_block_size / 2 < MIN_NZ_PER_BLOCK) {
      break;
    }
  }

  CUEMBED_ASSERT(nz_block_size >= MIN_NZ_PER_BLOCK);

  const int smem_per_nz_block =
      GetSmemBytesPerNzBlock<IndexT, WeightT>(nz_block_size);

  int nz_blocks_per_cta = 1;
  if (threads_per_nonzero <= DEFAULT_THREADS_PER_CTA) {
    nz_blocks_per_cta = DEFAULT_THREADS_PER_CTA / threads_per_nonzero;
  }

  while ((nz_blocks_per_cta * smem_per_nz_block) >= MAX_SHMEM_BYTES_PER_CTA) {
    nz_blocks_per_cta /= 2;
  }

  CUEMBED_ASSERT(nz_blocks_per_cta > 0);

  return std::make_tuple(
      element_per_load, nz_blocks_per_cta, nz_block_size, smem_per_nz_block);
}

/**
 * @brief Embedding backward propagation function. Handles full and compressed
 * gradients.
 *
 * @tparam GradT Gradient datatype
 * @tparam IndexT Index datatype
 * @tparam WeightT Weight datatype
 *
 * @param grad_y Pointer to the incoming gradient.
 * @param embed_width Number of elements in each embedding row.
 * @param num_grad_embedding_rows Number of rows in grad_embedding.
 * @param nnz  Total number of indices in COO input.
 * @param transpose_indices Pointer to the transposed lookup indices.
 * @param transpose_sample_ids Pointer to the transposed sample IDs.
 * @param transpose_remapped_indices Pointer to the remapped lookup indices
 * (i.e. from ComputeCompressedGradIndices), required only if computing
 * compressed gradient.
 * @param transpose_weights Pointer to the weight array. Set to nullptr for
 * unweighted.
 * @param skip_grad_init If true, skip zero-initializion of grad_embedding.
 * @param grad_embedding Pointer to the gradient wrt embedding table.
 * @param inverse_mapping Pointer to the table indices corresponding to each row
 * in grad_embedding, produced only for compressed gradients.
 * @param stream Optional. The cudaStream to launch the kernel asynchronously.
 * If not specified, will launch the kernel on default stream.
 */
template <typename GradT, typename IndexT>
void EmbeddingBackward(const GradT* grad_y,
                       const int embed_width,
                       const int num_grad_embedding_rows,
                       const int nnz,
                       const IndexT* transpose_indices,
                       const IndexT* transpose_sample_ids,
                       const IndexT* transpose_remapped_indices,
                       const GradT* transpose_weights,
                       const bool skip_grad_init,
                       GradT* grad_embedding,
                       IndexT* inverse_mapping,
                       const cudaStream_t stream = 0) {
  using WeightT = GradT;

  // Choose which indices to reference in the kernel
  const IndexT* transpose_indices_ = transpose_remapped_indices
                                         ? transpose_remapped_indices
                                         : transpose_indices;

  // Write mapping from sparse to dense indices
  if (transpose_remapped_indices) {
    const int nthreads = DEFAULT_THREADS_PER_CTA;
    CompactSparseIndicesKernel<IndexT>
        <<<(nnz + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
            transpose_indices,
            transpose_remapped_indices,
            inverse_mapping,
            nnz);
  }

  // Initialize gradient to zero
  if (!skip_grad_init) {
    cudaMemsetAsync(
        grad_embedding,
        0,
        (int64_t)num_grad_embedding_rows * (int64_t)embed_width * sizeof(GradT),
        stream);
  }

  auto [element_per_load, nz_blocks_per_cta, nz_block_size, smem_per_nz_block] =
      GetGradKernelLaunchParams<GradT, IndexT, WeightT>(embed_width, nnz);

  // Execute COO backward kernel
  const int nz_per_cta = nz_block_size * nz_blocks_per_cta;
  dim3 launch_grid((nnz + nz_per_cta - 1) / nz_per_cta, 1, 1);
  dim3 launch_block(embed_width / element_per_load, nz_blocks_per_cta, 1);
  size_t smem_size = nz_blocks_per_cta * smem_per_nz_block;

  if (element_per_load == 8) {
    GRAD_EMBEDDING_LOOKUP_DISPATCH(8);
  } else if (element_per_load == 4) {
    GRAD_EMBEDDING_LOOKUP_DISPATCH(4);
  } else if (element_per_load == 2) {
    GRAD_EMBEDDING_LOOKUP_DISPATCH(2);
  } else if (element_per_load == 1) {
    GRAD_EMBEDDING_LOOKUP_DISPATCH(1);
  } else {
    CUEMBED_ASSERT(0);
  }
}

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_EMBEDDING_LOOKUP_CUH_
