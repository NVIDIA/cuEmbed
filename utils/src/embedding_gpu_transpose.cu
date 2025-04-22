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
#include "cuembed/include/index_transforms.cuh"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace utils {

template <typename IndexT, typename OffsetT, typename WeightT>
void RunTranspose(const utils::AllocationOptions& options,
                  const thrust::device_vector<IndexT>& indices,
                  const thrust::device_vector<OffsetT>& offsets,
                  const thrust::device_vector<WeightT>& weights,
                  const OffsetT nnz,
                  thrust::device_vector<IndexT>* transpose_indices,
                  thrust::device_vector<IndexT>* transpose_remapped_indices,
                  thrust::device_vector<IndexT>* transpose_sample_ids,
                  thrust::device_vector<WeightT>* transpose_weights,
                  thrust::device_vector<IndexT>* sample_ids,
                  thrust::device_vector<char>* transpose_workspace) {
  if (options.combine_mode() == CombineMode::kConcat) {
    ExtractRowIdsForConcat<IndexT>(nnz, sample_ids->data().get());
  } else if (options.is_csr()) {
    ExtractRowIdsFromCSR<IndexT, OffsetT>(
        offsets.data().get(), options.batch_size(), sample_ids->data().get());
  } else {
    ExtractRowIdsFromFixed<IndexT>(
        options.batch_size(), options.hotness(), sample_ids->data().get());
  }

  const WeightT* weight_ptr = nullptr;
  WeightT* transpose_weight_ptr = nullptr;
  if (options.is_weighted()) {
    weight_ptr = weights.data().get();
    transpose_weight_ptr = transpose_weights->data().get();
  }

  size_t lwork = transpose_workspace->size();
  Transpose<IndexT, WeightT>(sample_ids->data().get(),
                             indices.data().get(),
                             weight_ptr,
                             nnz,
                             transpose_indices->data().get(),
                             transpose_sample_ids->data().get(),
                             transpose_weight_ptr,
                             transpose_workspace->data().get(),
                             &lwork);

  if (options.compressed_grad()) {
    ComputeCompressedGradIndices<IndexT>(
        transpose_indices->data().get(),
        nnz,
        transpose_remapped_indices->data().get(),
        transpose_workspace->data().get(),
        &lwork);
  }
}

#define RUN_TRANSPOSE_TEMPLATE(IndexT, OffsetT, WeightT)         \
  template void RunTranspose<IndexT, OffsetT, WeightT>(          \
      const utils::AllocationOptions& options,                   \
      const thrust::device_vector<IndexT>& indices,              \
      const thrust::device_vector<OffsetT>& offsets,             \
      const thrust::device_vector<WeightT>& weights,             \
      const OffsetT nnz,                                         \
      thrust::device_vector<IndexT>* transpose_indices,          \
      thrust::device_vector<IndexT>* transpose_remapped_indices, \
      thrust::device_vector<IndexT>* transpose_sample_ids,       \
      thrust::device_vector<WeightT>* transpose_weights,         \
      thrust::device_vector<IndexT>* sample_ids,                 \
      thrust::device_vector<char>* transpose_workspace);

RUN_TRANSPOSE_TEMPLATE(int32_t, int, float);
RUN_TRANSPOSE_TEMPLATE(int64_t, int, float);
RUN_TRANSPOSE_TEMPLATE(int32_t, int, __half);
RUN_TRANSPOSE_TEMPLATE(int64_t, int, __half);

#undef RUN_TRANSPOSE_TEMPLATE

}  // namespace utils

}  // namespace cuembed
