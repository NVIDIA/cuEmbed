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

#ifndef UTILS_INCLUDE_EMBEDDING_LOOKUP_CPU_HPP_
#define UTILS_INCLUDE_EMBEDDING_LOOKUP_CPU_HPP_

#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cuembed/include/embedding_lookup_types.cuh"

namespace cuembed {

template <typename InputT,
          typename OutputT,
          typename IndexT,
          typename OffsetT,
          bool fp16_math>
void EmbeddingForwardCpu(const InputT* params,
                         const int embed_width,
                         const int batch_size,
                         const int num_hots,
                         const IndexT* indices,
                         const OffsetT* offsets,
                         const GetElemT<InputT>* weights,
                         OutputT* ret,
                         const CombineMode reduce) {
  using ElemT = GetElemT<InputT>;
  // Weights can only be sum.
  CHECK(weights == nullptr || reduce == CombineMode::kSum);
  // CSR or fixed hotness.
  CHECK((offsets != nullptr && num_hots == 0) ||
        (offsets == nullptr && num_hots > 0));
  // CSR does not support concat.
  CHECK(offsets == nullptr || reduce != CombineMode::kConcat);
  for (int i = 0; i < batch_size; ++i) {
    for (int k = 0; k < embed_width; ++k) {
      using SumT = typename std::conditional<fp16_math, ElemT, float>::type;
      SumT sum = 0.0f;
      int hotness =
          (offsets == nullptr) ? num_hots : (offsets[i + 1] - offsets[i]);
      int index_start = (offsets == nullptr) ? i * num_hots : offsets[i];
      IndexT write_idx = i * embed_width + k;
      for (int j = 0; j < hotness; ++j) {
        int64_t read_idx =
            static_cast<int64_t>(indices[index_start + j]) * embed_width + k;
        if (reduce == CombineMode::kConcat) {
          write_idx = index_start * embed_width + j * embed_width + k;
          ret[write_idx] = params[read_idx];
        } else if (reduce == CombineMode::kSum ||
                   reduce == CombineMode::kMean) {
          ElemT weight = (weights == nullptr) ? static_cast<ElemT>(1.0f)
                                              : weights[index_start + j];
          sum += VecCast<SumT, ElemT>(params[read_idx]) * weight;
        } else {
          CHECK(false) << "reduce type not supported.";
        }
      }
      if (reduce == CombineMode::kSum) {
        ret[write_idx] = VecCast<OutputT, SumT>(sum);
      } else if (reduce == CombineMode::kMean) {
        if (hotness == 0) {
          // Preserve sign.
          ret[write_idx] =
              VecCast<OutputT, SumT>(sum * static_cast<SumT>(0.0f));
        } else {
          ret[write_idx] =
              VecCast<OutputT, SumT>(sum * static_cast<SumT>(1.0f / hotness));
        }
      }
    }
  }
}

template <typename GradT, typename IndexT>
void EmbeddingBackwardCpu(const GradT* result_grad,
                          const int embed_width,
                          const int num_grad_embedding_rows,
                          const int nnz,
                          const IndexT* transpose_indices,
                          const IndexT* transpose_sample_ids,
                          const IndexT* transpose_remapped_indices,
                          const GradT* transpose_weights,
                          const bool skip_grad_init,
                          GradT* grad_embedding,
                          IndexT* inverse_mapping) {
  using WeightT = GradT;
  if (nnz == 0) return;
  if (transpose_remapped_indices != nullptr) {
    CHECK(grad_embedding != nullptr);
    CHECK(inverse_mapping != nullptr);

    // Set grad embedding indices
    inverse_mapping[0] = transpose_indices[0];
    int cnt = 1;
    for (int i = 1; i < nnz; i++) {
      if (transpose_remapped_indices[i - 1] != transpose_remapped_indices[i]) {
        inverse_mapping[cnt] = transpose_indices[i];
        cnt++;
      }
    }
  }
  if (!skip_grad_init) {
    memset(grad_embedding,
           0,
           (int64_t)num_grad_embedding_rows * (int64_t)embed_width *
               sizeof(GradT));
  }
  // Loop over nnz, load index and offset, then loop over embed dim
  for (int nz = 0; nz < nnz; nz++) {
    IndexT index = (transpose_remapped_indices != nullptr)
                       ? transpose_remapped_indices[nz]
                       : transpose_indices[nz];
    IndexT sample_id = transpose_sample_ids[nz];
    WeightT weight = (transpose_weights != nullptr)
                         ? transpose_weights[nz]
                         : static_cast<WeightT>(1.0f);
    for (int e = 0; e < embed_width; e++) {
      grad_embedding[e + index * embed_width] +=
          result_grad[e + sample_id * embed_width] * weight;
    }
  }
}

}  // namespace cuembed

#endif  // UTILS_INCLUDE_EMBEDDING_LOOKUP_CPU_HPP_
