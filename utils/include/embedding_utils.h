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

#ifndef UTILS_INCLUDE_EMBEDDING_UTILS_H_
#define UTILS_INCLUDE_EMBEDDING_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cuembed/include/embedding_lookup_types.cuh"
#include "utils/include/embedding_allocation.h"

#define CHECK_CUDA(cmd)                                                \
  do {                                                                 \
    cudaError_t e = cmd;                                               \
    if (e != cudaSuccess) {                                            \
      LOG(FATAL) << absl::StrFormat("Failed: Cuda error %s:%d '%s'\n", \
                                    __FILE__,                          \
                                    __LINE__,                          \
                                    cudaGetErrorString(e));            \
    }                                                                  \
  } while (0)

namespace cuembed {
namespace utils {
struct Near {
  float tolerance_;

  explicit Near(const float tol) : tolerance_(tol) {}

  __host__ __device__ bool operator()(const float& a, const float& b) const {
    return fabsf(a - b) <= tolerance_;
  }

  __host__ __device__ bool operator()(const __half& a, const __half& b) const {
    return fabsf(__half2float(a) - __half2float(b)) <= tolerance_;
  }
};

template <const char* test_name_str>
class EmbeddingRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    typedef typename T::EmbedType EmbedType;
    typedef typename T::IndexType IndexType;

    std::string test_name = std::string(test_name_str);
    if (std::is_same_v<EmbedType, float>) {
      test_name += std::string("Embed[float]");
    }
    if (std::is_same_v<EmbedType, __half>) {
      test_name += std::string("Embed[half]");
    }
    if (std::is_same_v<IndexType, int32_t>) {
      test_name += std::string("Index[int32]");
    }
    if (std::is_same_v<IndexType, int64_t>) {
      test_name += std::string("Index[int64]");
    }
    test_name += std::to_string(i);
    return test_name;
  }
};

template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void RunForward(const utils::AllocationOptions& options,
                const thrust::device_vector<ElemT>& embedding,
                const thrust::device_vector<IndexT>& indices,
                const thrust::device_vector<OffsetT>& offsets,
                const thrust::device_vector<ElemT>& weights,
                thrust::device_vector<ElemT>* result);

template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void RunForwardReference(const utils::AllocationOptions& options,
                         const thrust::universal_vector<ElemT>& embedding,
                         const thrust::universal_vector<IndexT>& indices,
                         const thrust::universal_vector<OffsetT>& offsets,
                         const thrust::universal_vector<ElemT>& weights,
                         thrust::universal_vector<ElemT>* result);

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
                  thrust::device_vector<char>* transpose_workspace);

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
    thrust::universal_vector<WeightT>* transpose_weights);

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
    thrust::device_vector<IndexT>* inverse_mapping);

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
    thrust::universal_vector<IndexT>* inverse_mapping);

}  // namespace utils
}  // namespace cuembed

#endif  // UTILS_INCLUDE_EMBEDDING_UTILS_H_
