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
#include "cuembed/include/embedding_lookup.cuh"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace utils {

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

#define RUN_FORWARD_TEMPLATE(ElemT, IndexT, OffsetT, fp16_math) \
  template void RunForward<ElemT, IndexT, OffsetT, fp16_math>(  \
      const utils::AllocationOptions& options,                  \
      const thrust::device_vector<ElemT>& embedding,            \
      const thrust::device_vector<IndexT>& indices,             \
      const thrust::device_vector<OffsetT>& offsets,            \
      const thrust::device_vector<ElemT>& weights,              \
      thrust::device_vector<ElemT>* result);

RUN_FORWARD_TEMPLATE(float, int32_t, int, false);
RUN_FORWARD_TEMPLATE(float, int64_t, int, false);
RUN_FORWARD_TEMPLATE(__half, int32_t, int, false);
RUN_FORWARD_TEMPLATE(__half, int64_t, int, false);
RUN_FORWARD_TEMPLATE(float, int32_t, int, true);
RUN_FORWARD_TEMPLATE(float, int64_t, int, true);
RUN_FORWARD_TEMPLATE(__half, int32_t, int, true);
RUN_FORWARD_TEMPLATE(__half, int64_t, int, true);

#undef RUN_FORWARD_TEMPLATE

}  // namespace utils

}  // namespace cuembed
