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
#ifndef CUEMBED_INCLUDE_EMBEDDING_LOOKUP_KERNELS_CUH_
#define CUEMBED_INCLUDE_EMBEDDING_LOOKUP_KERNELS_CUH_

#include "cuembed/include/embedding_lookup_ops.cuh"

// Parentheses after __launch_bounds__ confuses Doxygen. Make it macro
#define LAUNCH_BOUNDS_1024_1 __launch_bounds__(1024, 1)

//! cuEmbed main namespace
namespace cuembed {
/*!
 * \brief The actual implementation of the embedding lookup kernel.
 */
template <class IndexLoaderT,
          class AddresserT,
          class CombinerT,
          bool IsWeighted,
          int UnrollFactor>
__device__ __forceinline__ void EmbeddingLookupImpl(
    const typename AddresserT::InputType* __restrict__ params,
    const int embed_width,
    const int batch_size,
    const typename IndexLoaderT::IndexType* __restrict__ indices,
    const typename IndexLoaderT::OffsetType* __restrict__ offsets,
    const int num_hots,
    const typename IndexLoaderT::WeightType* __restrict__ weights,
    typename AddresserT::OutputType* __restrict__ ret) {
  const int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
  IndexLoaderT index_loader(
      batch_size, sample_id, indices, weights, offsets, num_hots);

  if (sample_id >= batch_size) {
    return;
  }
  CombinerT combiner;

  AddresserT addresser(params, ret, sample_id, num_hots, embed_width);

  int64_t embed_row_offset = threadIdx.x;
  int64_t output_row_offset = threadIdx.x;

#pragma unroll UnrollFactor
  for (int i = 0; i < index_loader.GetHotness(); ++i) {
    auto index = index_loader.GetLookUpIndex(i);
    if constexpr (IsWeighted) {
      auto weight = index_loader.GetWeight(i);
      combiner.Gather(addresser.GetEmbeddingAddress(index) + embed_row_offset,
                      weight);
    } else {
      combiner.Gather(addresser.GetEmbeddingAddress(index) + embed_row_offset);
    }
    combiner.OutputForConcatIfNeeded(addresser.GetConcatOutputAddress(i) +
                                     output_row_offset);
  }
  combiner.OutputForReductionIfNeeded(addresser.GetOutputAddress() +
                                      output_row_offset);
}

/*!
 * \brief Embedding lookup kernel for fixed hotness and CSR lookup index layout.
 *
 * Templatization for this kernel is only for the basic operations:
 * IndexLoaderT, AddresserT and CombinerT.
 *
 * Organization for Grid and Blocks:
 *   1. Each CTA is a 2D block of threads.
 *      Each CTA is responsible for processing multiple lookup samples.
 *      Threads with the same blockDim.y in the CTA are processing the same
 *      sample. To maximize loads in flight, each thread loads multiple samples
 *      per load.
 *      BlockDim.x = embed_width / elements_per_load.
 *      BlockDim.y = samples_per_cta.
 *   2. The grid is a 1D array of CTAs.
 *      GridDim.x = batch_size / samples_per_cta.
 *
 * The kernel contains the following operations:
 *   1. IndexLoader calculates the hotness and the index offset loads for a
 *      specific sample.
 *      For fixed hotness, IndexLoader lookup indices needed by the CTA into
 *      shared memory and calculates the offset based on the loaded indices.
 *      For CSR layout, IndexLoader just does the offset and hotness
 *      calculation, then loads the lookup index from global memory when the
 *      index is needed.
 *   2. Addresser calculates the lookup addresses based on the lookup indices.
 *   3. Combiner issues out parallel loads and writes output according to the
 *      calculated addresses.
 *
 * Different modes of reduction/concatenation are realized by template
 * specialization of the combiner.
 *
 * Different layouts of the lookup indices are separated by template
 * specialization of offsets_or_hotness parameter.
 *
 * For future integration of Embed Cache, we can templatize the addresser with
 * the cache to get the indirect mapping of the actual address.
 */
template <class IndexLoaderT,
          class AddresserT,
          class CombinerT,
          int UnrollFactor>
__global__ void LAUNCH_BOUNDS_1024_1 EmbeddingLookUpKernel(
    const typename AddresserT::InputType* __restrict__ params,
    const int embed_width,
    const int batch_size,
    const typename IndexLoaderT::IndexType* __restrict__ indices,
    const typename IndexLoaderT::OffsetType* __restrict__ offsets,
    const int num_hots,
    const typename IndexLoaderT::WeightType* __restrict__ weights,
    typename AddresserT::OutputType* __restrict__ ret) {
  EmbeddingLookupImpl<IndexLoaderT, AddresserT, CombinerT, true, UnrollFactor>(
      params,
      embed_width,
      batch_size,
      indices,
      offsets,
      num_hots,
      weights,
      ret);
}

/*!
 * \brief Explicit specialization of the embedding lookup kernel for the not
 * weighted use case.
 *
 * This specialization cuts down the number of registers needed for this kernel
 * and achieves higher occupancy.
 */
template <class IndexLoaderT,
          class AddresserT,
          class CombinerT,
          int UnrollFactor>
__global__ void LAUNCH_BOUNDS_1024_1 EmbeddingLookUpKernel(
    const typename AddresserT::InputType* __restrict__ params,
    const int embed_width,
    const int batch_size,
    const typename IndexLoaderT::IndexType* __restrict__ indices,
    const typename IndexLoaderT::OffsetType* __restrict__ offsets,
    const int num_hots,
    std::nullptr_t,
    typename AddresserT::OutputType* __restrict__ ret) {
  EmbeddingLookupImpl<IndexLoaderT, AddresserT, CombinerT, false, UnrollFactor>(
      params,
      embed_width,
      batch_size,
      indices,
      offsets,
      num_hots,
      nullptr,
      ret);
}

/*!
 * \brief The actual implementation of the embedding backward kernel.
 */
template <typename GradIndexLoaderT,
          typename GradAddresserT,
          typename GradCombinerT,
          bool IsWeighted>
__device__ __forceinline__ void EmbeddingBackwardImpl(
    const typename GradAddresserT::GradType* __restrict__ grad_y,
    const int embed_width,
    const typename GradIndexLoaderT::IndexType* __restrict__ transpose_indices,
    const typename GradIndexLoaderT::
        IndexType* __restrict__ transpose_sample_ids,
    const typename GradIndexLoaderT::WeightType* __restrict__ transpose_weights,
    const int nnz,
    const int nz_block_size,
    typename GradAddresserT::GradType* __restrict__ grad_embedding) {
  GradIndexLoaderT index_loader(transpose_indices,
                                transpose_sample_ids,
                                transpose_weights,
                                nnz,
                                nz_block_size);

  GradAddresserT addresser(grad_y, grad_embedding, embed_width);

  GradCombinerT combiner;

  const int embed_offset = threadIdx.x;

  // For each nonzero assigned to this block
  for (int i = 0; i < index_loader.GetBlockNnz(); i++) {
    int row = index_loader.GetIndex(i);
    int col = index_loader.GetSampleId(i);
    bool write_flag = index_loader.ShouldWrite(i);
    bool atomic_flag = index_loader.ShouldAtomic(i);

    if constexpr (IsWeighted) {
      auto weight = index_loader.GetWeight(i);
      combiner.Gather(addresser.GetGradResultAddress(col) + embed_offset,
                      weight);
    } else {
      combiner.Gather(addresser.GetGradResultAddress(col) + embed_offset);
    }
    combiner.WriteOrAtomic(
        addresser.GetGradEmbeddingAddress(row) + embed_offset,
        write_flag,
        atomic_flag);
  }
}

/*!
 * \brief Embedding backward kernel for compressed and full gradients.
 *
 * Organization for Grid and Blocks:
 *   1. Each CTA is a 2D block of threads.
 *      Each CTA is responsible for processing multiple indices.
 *      Threads with the same blockDim.y in the CTA are processing the same
 *      indices.
 *      BlockDim.x = embed_width / elements_per_load.
 *      BlockDim.y = nonzero_blocks_per_cta
 *   2. The grid is a 1D array of CTAs.
 *      GridDim.x = nnz / nz_block_size / nonzero_blocks_per_cta.
 *
 */
template <typename GradIndexLoaderT,
          typename GradAddresserT,
          typename GradCombinerT>
__global__ void EmbeddingBackwardKernel(
    const typename GradAddresserT::GradType* __restrict__ grad_y,
    const int embed_width,
    const typename GradIndexLoaderT::IndexType* __restrict__ transpose_indices,
    const typename GradIndexLoaderT::
        IndexType* __restrict__ transpose_sample_ids,
    const typename GradIndexLoaderT::WeightType* __restrict__ transpose_weights,
    const int nnz,
    const int nz_block_size,
    typename GradAddresserT::GradType* __restrict__ grad_embedding) {
  EmbeddingBackwardImpl<GradIndexLoaderT, GradAddresserT, GradCombinerT, true>(
      grad_y,
      embed_width,
      transpose_indices,
      transpose_sample_ids,
      transpose_weights,
      nnz,
      nz_block_size,
      grad_embedding);
}

/*!
 * \brief Specialization of mbedding backward kernel for unweighted case
 *
 */
template <typename GradIndexLoaderT,
          typename GradAddresserT,
          typename GradCombinerT>
__global__ void EmbeddingBackwardKernel(
    const typename GradAddresserT::GradType* __restrict__ grad_y,
    const int embed_width,
    const typename GradIndexLoaderT::IndexType* __restrict__ transpose_indices,
    const typename GradIndexLoaderT::
        IndexType* __restrict__ transpose_sample_ids,
    std::nullptr_t,
    const int nnz,
    const int nz_block_size,
    typename GradAddresserT::GradType* __restrict__ grad_embedding) {
  EmbeddingBackwardImpl<GradIndexLoaderT, GradAddresserT, GradCombinerT, false>(
      grad_y,
      embed_width,
      transpose_indices,
      transpose_sample_ids,
      nullptr,
      nnz,
      nz_block_size,
      grad_embedding);
}

// Generate sparse indices for sparse output gradients
template <typename IndexT>
__global__ void CompactSparseIndicesKernel(const IndexT* indices,
                                           const IndexT* remapped_indices,
                                           IndexT* compacted_indices,
                                           const int nnz) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nnz) return;

  // If you're the first occurance, then write
  if ((tid == 0) || (indices[tid - 1] != indices[tid])) {
    IndexT cidx = remapped_indices[tid];
    compacted_indices[cidx] = indices[tid];
  }
}

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_EMBEDDING_LOOKUP_KERNELS_CUH_
