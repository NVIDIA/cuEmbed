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
#ifndef CUEMBED_INCLUDE_INDEX_TRANSFORMS_KERNELS_CUH_
#define CUEMBED_INCLUDE_INDEX_TRANSFORMS_KERNELS_CUH_

//! cuEmbed main namespace
namespace cuembed {

// Create expanded COO Offsets from CSR Offsets
template <typename OffsetT, typename IndexT>
__global__ void ExtractRowIdsFromCSRKernel(const OffsetT* offsets,
                                           IndexT* row_ids) {
  const int b = blockIdx.x;
  OffsetT start = offsets[b];
  OffsetT end = offsets[b + 1];
  for (OffsetT i = start + threadIdx.x; i < end; i += blockDim.x) {
    row_ids[i] = static_cast<IndexT>(b);
  }
}

// Create offsets from sequence
template <typename IndexT>
__global__ void ExtractSequenceKernel(const int nnz,
                                      const int int_div,
                                      IndexT* row_ids) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nnz) {
    row_ids[tid] = static_cast<IndexT>(tid / int_div);
  }
}

template <typename IndexT, typename WeightT>
struct WeightTuple {
  IndexT idx;
  WeightT weight;
};

template <typename IndexT, typename WeightT>
__global__ void PackToTuple(const IndexT* __restrict__ indices,
                            const WeightT* __restrict__ weights,
                            const int nnz,
                            WeightTuple<IndexT, WeightT>* __restrict__ vals) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nnz) {
    WeightTuple<IndexT, WeightT> t;
    t.idx = indices[tid];
    t.weight = weights[tid];
    vals[tid] = t;
  }
}

template <typename IndexT, typename WeightT>
__global__ void ExtractFromTuple(
    const WeightTuple<IndexT, WeightT>* __restrict__ vals,
    const int nnz,
    IndexT* __restrict__ indices,
    WeightT* __restrict__ weights) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nnz) {
    indices[tid] = vals[tid].idx;
    weights[tid] = vals[tid].weight;
  }
}

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_INDEX_TRANSFORMS_KERNELS_CUH_
