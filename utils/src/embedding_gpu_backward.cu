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

#include <thrust/device_vector.h>

#include "absl/log/check.h"
#include "cuembed/include/embedding_lookup.cuh"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace utils {

template <typename ElemT, typename IndexT, typename OffsetT>
void RunBackward(
    const utils::AllocationOptions& options,
    const thrust::device_vector<ElemT>& grad_y,
    const thrust::device_vector<IndexT>& transpose_indices,
    const thrust::device_vector<IndexT>& transpose_remapped_indices,
    const thrust::device_vector<IndexT>& transpose_sample_ids,
    const thrust::device_vector<ElemT>& transpose_weights,
    const thrust::device_vector<OffsetT>& offsets,
    const OffsetT nnz,
    const OffsetT num_unique,
    thrust::device_vector<ElemT>* grad_embedding,
    thrust::device_vector<IndexT>* inverse_mapping) {
  const IndexT* transpose_remapped_indices_ptr = nullptr;
  IndexT* inverse_mapping_ptr = nullptr;
  if (options.compressed_grad()) {
    transpose_remapped_indices_ptr = transpose_remapped_indices.data().get();
    inverse_mapping_ptr = inverse_mapping->data().get();
  }

  const ElemT* transpose_weights_ptr = nullptr;
  if (options.is_weighted()) {
    transpose_weights_ptr = transpose_weights.data().get();
  }

  cuembed::EmbeddingBackward<ElemT, IndexT>(
      grad_y.data().get(),
      options.embed_width(),
      options.compressed_grad() ? num_unique : options.num_categories(),
      nnz,
      transpose_indices.data().get(),
      transpose_sample_ids.data().get(),
      transpose_remapped_indices_ptr,
      transpose_weights_ptr,
      options.skip_grad_init(),
      grad_embedding->data().get(),
      inverse_mapping_ptr);
}

#define RUN_BACKWARD_TEMPLATE(GradT, IndexT, OffsetT)                  \
  template void RunBackward<GradT, IndexT, OffsetT>(                   \
      const utils::AllocationOptions& options,                         \
      const thrust::device_vector<GradT>& grad_y,                      \
      const thrust::device_vector<IndexT>& transpose_indices,          \
      const thrust::device_vector<IndexT>& transpose_remapped_indices, \
      const thrust::device_vector<IndexT>& transpose_sample_ids,       \
      const thrust::device_vector<GradT>& transpose_weights,           \
      const thrust::device_vector<OffsetT>& offsets,                   \
      const OffsetT nnz,                                               \
      const OffsetT num_unique,                                        \
      thrust::device_vector<GradT>* grad_embedding,                    \
      thrust::device_vector<IndexT>* inverse_mapping);

RUN_BACKWARD_TEMPLATE(float, int32_t, int);
RUN_BACKWARD_TEMPLATE(float, int64_t, int);
RUN_BACKWARD_TEMPLATE(__half, int32_t, int);
RUN_BACKWARD_TEMPLATE(__half, int64_t, int);

#undef RUN_BACKWARD_TEMPLATE

}  // namespace utils

}  // namespace cuembed
