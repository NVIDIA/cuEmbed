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

#include "absl/log/check.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// CPU reference implementations
#include "utils/include/embedding_lookup_cpu.hpp"
#include "utils/include/index_transforms_cpu.hpp"

namespace cuembed {

namespace utils {

template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void RunForwardReference(const utils::AllocationOptions& options,
                         const thrust::universal_vector<ElemT>& embedding,
                         const thrust::universal_vector<IndexT>& indices,
                         const thrust::universal_vector<OffsetT>& offsets,
                         const thrust::universal_vector<ElemT>& weights,
                         thrust::universal_vector<ElemT>* result) {
  const OffsetT* offsets_ptr = nullptr;
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
  EmbeddingForwardCpu<InputT, OutputT, IndexT, OffsetT, fp16_math>(
      embedding.data().get(),
      options.embed_width(),
      options.batch_size(),
      hotness,
      indices.data().get(),
      offsets_ptr,
      weight_ptr,
      result->data().get(),
      options.combine_mode());
}

#define RUN_FORWARD_TEMPLATE(ElemT, IndexT, OffsetT, fp16_math)         \
  template void RunForwardReference<ElemT, IndexT, OffsetT, fp16_math>( \
      const utils::AllocationOptions& options,                          \
      const thrust::universal_vector<ElemT>& embedding,                 \
      const thrust::universal_vector<IndexT>& indices,                  \
      const thrust::universal_vector<OffsetT>& offsets,                 \
      const thrust::universal_vector<ElemT>& weights,                   \
      thrust::universal_vector<ElemT>* result);

RUN_FORWARD_TEMPLATE(float, int32_t, int, false);
RUN_FORWARD_TEMPLATE(float, int64_t, int, false);
RUN_FORWARD_TEMPLATE(__half, int32_t, int, false);
RUN_FORWARD_TEMPLATE(__half, int64_t, int, false);
RUN_FORWARD_TEMPLATE(float, int32_t, int, true);
RUN_FORWARD_TEMPLATE(float, int64_t, int, true);
RUN_FORWARD_TEMPLATE(__half, int32_t, int, true);
RUN_FORWARD_TEMPLATE(__half, int64_t, int, true);

#undef RUN_FORWARD_TEMPLATE

template <typename IndexT, typename OffsetT, typename WeightT>
void RunTransposeReference(
    const utils::AllocationOptions& options,
    const thrust::universal_vector<IndexT>& indices,
    const thrust::universal_vector<OffsetT>& offsets,
    const thrust::universal_vector<WeightT>& weights,
    const int nnz,
    thrust::universal_vector<IndexT>* transpose_indices,
    thrust::universal_vector<IndexT>* transpose_remapped_indices,
    thrust::universal_vector<IndexT>* transpose_sample_ids,
    thrust::universal_vector<WeightT>* transpose_weights) {
  // Extract rows
  thrust::universal_vector<IndexT> sample_ids(indices.size(), 0);
  if (options.combine_mode() == CombineMode::kConcat) {
    ExtractRowIdsForConcatCpu<IndexT>(nnz, sample_ids.data().get());
  } else if (options.is_csr()) {
    ExtractRowIdsFromCSRCpu<IndexT, OffsetT>(
        offsets.data().get(), options.batch_size(), sample_ids.data().get());
  } else {
    ExtractRowIdsFromFixedCpu<IndexT>(
        options.batch_size(), options.hotness(), sample_ids.data().get());
  }

  const WeightT* weight_ptr = nullptr;
  WeightT* transpose_weight_ptr = nullptr;
  if (options.is_weighted()) {
    weight_ptr = weights.data().get();
    transpose_weight_ptr = transpose_weights->data().get();
  }

  TransposeCpu<IndexT, WeightT>(sample_ids.data().get(),
                                indices.data().get(),
                                weight_ptr,
                                nnz,
                                transpose_indices->data().get(),
                                transpose_sample_ids->data().get(),
                                transpose_weight_ptr);

  // Compute sparse indices
  if (options.compressed_grad()) {
    ComputeCompressedGradIndicesCpu<IndexT>(
        transpose_indices->data().get(),
        nnz,
        transpose_remapped_indices->data().get());
  }
}

#define RUN_TRANSPOSE_TEMPLATE(IndexT, OffsetT, WeightT)            \
  template void RunTransposeReference<IndexT, OffsetT, WeightT>(    \
      const utils::AllocationOptions& options,                      \
      const thrust::universal_vector<IndexT>& indices,              \
      const thrust::universal_vector<OffsetT>& offsets,             \
      const thrust::universal_vector<WeightT>& weights,             \
      const int nnz,                                                \
      thrust::universal_vector<IndexT>* transpose_indices,          \
      thrust::universal_vector<IndexT>* transpose_remapped_indices, \
      thrust::universal_vector<IndexT>* transpose_sample_ids,       \
      thrust::universal_vector<WeightT>* transpose_weights);

RUN_TRANSPOSE_TEMPLATE(int32_t, int, float);
RUN_TRANSPOSE_TEMPLATE(int64_t, int, float);
RUN_TRANSPOSE_TEMPLATE(int32_t, int, __half);
RUN_TRANSPOSE_TEMPLATE(int64_t, int, __half);

#undef RUN_TRANSPOSE_TEMPLATE

template <typename ElemT, typename IndexT, typename OffsetT>
void RunBackwardReference(
    const utils::AllocationOptions& options,
    const thrust::universal_vector<ElemT>& grad_y,
    const thrust::universal_vector<IndexT>& transpose_indices,
    const thrust::universal_vector<IndexT>& transpose_remapped_indices,
    const thrust::universal_vector<IndexT>& transpose_sample_ids,
    const thrust::universal_vector<ElemT>& transpose_weights,
    const thrust::universal_vector<OffsetT>& offsets,
    const int nnz,
    thrust::universal_vector<ElemT>* grad_embedding,
    thrust::universal_vector<IndexT>* inverse_mapping) {
  const ElemT* transpose_weight_ptr = nullptr;
  if (options.is_weighted()) {
    transpose_weight_ptr = transpose_weights.data().get();
  }
  const IndexT* transpose_remapped_indices_ptr = nullptr;
  IndexT* inverse_mapping_ptr = nullptr;
  int num_grad_embedding_rows = options.num_categories();
  if (options.compressed_grad()) {
    transpose_remapped_indices_ptr = transpose_remapped_indices.data().get();
    inverse_mapping_ptr = inverse_mapping->data().get();
    num_grad_embedding_rows = transpose_remapped_indices[nnz - 1] + 1;
  }

  EmbeddingBackwardCpu<ElemT, IndexT>(grad_y.data().get(),
                                      options.embed_width(),
                                      num_grad_embedding_rows,
                                      nnz,
                                      transpose_indices.data().get(),
                                      transpose_sample_ids.data().get(),
                                      transpose_remapped_indices_ptr,
                                      transpose_weight_ptr,
                                      options.skip_grad_init(),
                                      grad_embedding->data().get(),
                                      inverse_mapping_ptr);
}

#define RUN_BACKWARD_TEMPLATE(GradT, IndexT, OffsetT)                     \
  template void RunBackwardReference<GradT, IndexT, OffsetT>(             \
      const utils::AllocationOptions& options,                            \
      const thrust::universal_vector<GradT>& grad_y,                      \
      const thrust::universal_vector<IndexT>& transpose_indices,          \
      const thrust::universal_vector<IndexT>& transpose_remapped_indices, \
      const thrust::universal_vector<IndexT>& transpose_sample_ids,       \
      const thrust::universal_vector<GradT>& transpose_weights,           \
      const thrust::universal_vector<OffsetT>& offsets,                   \
      const int nnz,                                                      \
      thrust::universal_vector<GradT>* grad_embedding,                    \
      thrust::universal_vector<IndexT>* inverse_mapping);

RUN_BACKWARD_TEMPLATE(float, int32_t, int);
RUN_BACKWARD_TEMPLATE(float, int64_t, int);
RUN_BACKWARD_TEMPLATE(__half, int32_t, int);
RUN_BACKWARD_TEMPLATE(__half, int64_t, int);

#undef RUN_BACKWARD_TEMPLATE

}  // namespace utils

}  // namespace cuembed
