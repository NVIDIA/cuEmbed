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

#ifndef UTILS_INCLUDE_EMBEDDING_ALLOCATION_H_
#define UTILS_INCLUDE_EMBEDDING_ALLOCATION_H_

#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "cuembed/include/embedding_lookup_types.cuh"

namespace cuembed {

// String literals live in cuembed namespace for ease of use.
// A string literal for million. E.g., num_categories = 10_M.
constexpr unsigned long long operator"" _M(  // NOLINT
    unsigned long long int val) {            // NOLINT
  return val * 1024 * 1024;
}

// A string literal for thousand. E.g., batch_size = 64_K.
constexpr unsigned long long operator"" _K(  // NOLINT
    unsigned long long int val) {            // NOLINT
  return val * 1024;
}

namespace utils {

// A wrapper class for different options.
class AllocationOptions {
 public:
  AllocationOptions() {}

  // Setters and getters of each allocation option.
  AllocationOptions& num_categories(int num_categories);
  int32_t num_categories() const { return num_categories_; }

  AllocationOptions& batch_size(int32_t batch_size);
  int32_t batch_size() const { return batch_size_; }

  AllocationOptions& hotness(int32_t hotness);
  int32_t hotness() const { return hotness_; }

  AllocationOptions& alpha(float alpha);
  float alpha() const { return alpha_; }

  AllocationOptions& combine_mode(CombineMode type);
  CombineMode combine_mode() const { return combine_mode_; }

  AllocationOptions& embed_width(int32_t embed_width);
  int32_t embed_width() const { return embed_width_; }

  AllocationOptions& permute_indices(bool permute_indices);
  bool permute_indices() const { return permute_indices_; }

  AllocationOptions& shuffle_indices(int32_t shuffle_indices);
  bool shuffle_indices() const { return shuffle_indices_; }

  AllocationOptions& is_csr(bool is_csr);
  bool is_csr() const { return is_csr_; }

  AllocationOptions& is_weighted(bool is_weighted);
  bool is_weighted() const { return is_weighted_; }

  AllocationOptions& compressed_grad(bool compressed_grad);
  bool compressed_grad() const { return compressed_grad_; }

  AllocationOptions& skip_grad_init(bool skip_grad_init);
  bool skip_grad_init() const { return skip_grad_init_; }

 private:
  int32_t num_categories_{0};
  int32_t batch_size_{0};
  int32_t hotness_{0};
  float alpha_{0};
  int32_t embed_width_{0};
  bool permute_indices_{true};
  bool shuffle_indices_{true};
  bool is_csr_{false};
  bool is_weighted_{false};
  bool compressed_grad_{false};
  bool skip_grad_init_{false};
  CombineMode combine_mode_{CombineMode::kSum};
};

template <typename InputT,
          typename IndexT,
          typename OffsetT,
          typename WeightT,
          typename OutputT,
          typename GradT>
struct UniversalEmbeddingAllocation {
  thrust::universal_vector<InputT> embedding;
  thrust::universal_vector<IndexT> indices;
  thrust::universal_vector<OffsetT> offsets;
  thrust::universal_vector<WeightT> weights;
  thrust::universal_vector<OutputT> result;
  thrust::universal_vector<IndexT> transpose_indices;
  thrust::universal_vector<IndexT> transpose_remapped_indices;
  thrust::universal_vector<IndexT> transpose_sample_ids;
  thrust::universal_vector<WeightT> transpose_weights;
  thrust::universal_vector<IndexT> sample_ids;
  thrust::universal_vector<char> transpose_workspace;
  thrust::universal_vector<GradT> grad_y;
  thrust::universal_vector<GradT> grad_embedding;
  thrust::universal_vector<IndexT> inverse_mapping;
};

template <typename InputT,
          typename IndexT,
          typename OffsetT,
          typename WeightT,
          typename OutputT,
          typename GradT>
struct DeviceEmbeddingAllocation {
  thrust::device_vector<InputT> embedding;
  thrust::device_vector<IndexT> indices;
  thrust::device_vector<OffsetT> offsets;
  thrust::device_vector<WeightT> weights;
  thrust::device_vector<OutputT> result;
  thrust::device_vector<IndexT> transpose_indices;
  thrust::device_vector<IndexT> transpose_remapped_indices;
  thrust::device_vector<IndexT> transpose_sample_ids;
  thrust::device_vector<IndexT> sample_ids;
  thrust::device_vector<WeightT> transpose_weights;
  thrust::device_vector<char> transpose_workspace;
  thrust::device_vector<GradT> grad_y;
  thrust::device_vector<GradT> grad_embedding;
  thrust::device_vector<IndexT> inverse_mapping;
};

template <typename InputT,
          typename IndexT,
          typename OffsetT,
          typename WeightT,
          typename OutputT,
          typename GradT>
void AllocateHost(const AllocationOptions& options,
                  UniversalEmbeddingAllocation<InputT,
                                               IndexT,
                                               OffsetT,
                                               WeightT,
                                               OutputT,
                                               GradT>* allocation,
                  bool forward_only = false);

template <typename InputT,
          typename IndexT,
          typename OffsetT,
          typename WeightT,
          typename OutputT,
          typename GradT>
void AllocateDevice(
    const AllocationOptions& options,
    const UniversalEmbeddingAllocation<InputT,
                                       IndexT,
                                       OffsetT,
                                       WeightT,
                                       OutputT,
                                       GradT>& universal_allocation,
    DeviceEmbeddingAllocation<InputT, IndexT, OffsetT, WeightT, OutputT, GradT>*
        device_allocation,
    bool forward_only = false);

}  // namespace utils
}  // namespace cuembed

#endif  // UTILS_INCLUDE_EMBEDDING_ALLOCATION_H_
