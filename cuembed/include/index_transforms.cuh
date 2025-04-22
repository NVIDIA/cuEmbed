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
#ifndef CUEMBED_INCLUDE_INDEX_TRANSFORMS_CUH_
#define CUEMBED_INCLUDE_INDEX_TRANSFORMS_CUH_

// clang-format off
#include <cub/cub.cuh>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "cuembed/include/embedding_lookup.cuh"
#include "cuembed/include/index_transforms_kernels.cuh"
// clang-format on

namespace cuembed {

/*!
 * \brief Produce a nnz-length row_ids array which has fixed offsets from 0 to
 * batch_size, e.g.: num_hots = 3 -> row_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
 *
 *  Requires a workspace query in which the function is called first with the
 *  "work" parameter set to NULL. The required temporary memory size in bytes
 *  will be returned in the lwork parameter by reference.
 *
 */
template <typename IndexT>
void ExtractRowIdsFromFixed(const int batch_size,
                            const int num_hots,
                            IndexT* row_ids,
                            const cudaStream_t stream = 0) {
  const int nnz = batch_size * num_hots;
  const int nthreads = DEFAULT_THREADS_PER_CTA;
  ExtractSequenceKernel<IndexT>
      <<<(nnz + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
          nnz, num_hots, row_ids);
}

/*!
 * \brief Produce a nnz-length row_ids array which has explicit offsets read
 * from CSR, e.g.: offsets : [0, 2, 3, 5] -> row_ids = [0, 0, 1, 2, 2]
 *
 *  Requires a workspace query in which the function is called first with the
 *  "work" parameter set to NULL. The required temporary memory size in bytes
 *  will be returned in the lwork parameter by reference.
 *
 */
template <typename IndexT, typename OffsetT>
void ExtractRowIdsFromCSR(const OffsetT* offsets,
                          const int batch_size,
                          IndexT* row_ids,
                          const cudaStream_t stream = 0) {
  const int nthreads = DEFAULT_THREADS_PER_CTA;
  ExtractRowIdsFromCSRKernel<OffsetT, IndexT>
      <<<batch_size, nthreads, 0, stream>>>(offsets, row_ids);
}

/*!
 * \brief Produce a nnz-length row_ids array which has the sequence 0 .. nnz.
 * e.g. row_ids = [0, 1, 2, 3, ...]
 *
 *  Requires a workspace query in which the function is called first with the
 *  "work" parameter set to NULL. The required temporary memory size in bytes
 *  will be returned in the lwork parameter by reference.
 *
 */
template <typename IndexT>
void ExtractRowIdsForConcat(const int nnz,
                            IndexT* row_ids,
                            const cudaStream_t stream = 0) {
  const int nthreads = DEFAULT_THREADS_PER_CTA;
  ExtractSequenceKernel<IndexT>
      <<<(nnz + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
          nnz, 1, row_ids);
}

template <typename IndexT>
void TransposeUnweighted(const IndexT* rows,
                         const IndexT* cols,
                         const int nnz,
                         IndexT* transpose_rows,
                         IndexT* transpose_cols,
                         char* work,
                         size_t* lwork,
                         const cudaStream_t stream = 0) {
  void* nullwork = nullptr;
  size_t sort_bytes = 0;
  const int begin_bit = 0;
  const int end_bit = sizeof(IndexT) * 8;
  cub::DeviceRadixSort::SortPairs(nullwork,
                                  sort_bytes,
                                  cols,
                                  transpose_rows,
                                  rows,
                                  transpose_cols,
                                  nnz,
                                  begin_bit,
                                  end_bit,
                                  stream);

  size_t required_workspace = sort_bytes;

  if (work == nullptr) {
    *lwork = required_workspace;
    return;
  }

  assert(*lwork >= required_workspace);
  cub::DeviceRadixSort::SortPairs(static_cast<void*>(work),
                                  *lwork,
                                  cols,
                                  transpose_rows,
                                  rows,
                                  transpose_cols,
                                  nnz,
                                  begin_bit,
                                  end_bit,
                                  stream);
}

template <typename IndexT, typename WeightT>
void TransposeWeighted(const IndexT* rows,
                       const IndexT* cols,
                       const WeightT* weights,
                       const int nnz,
                       IndexT* transpose_rows,
                       IndexT* transpose_cols,
                       WeightT* transpose_weights,
                       char* work,
                       size_t* lwork,
                       const cudaStream_t stream = 0) {
  size_t buffer_bytes = 2 * nnz * sizeof(WeightTuple<IndexT, WeightT>);

  WeightTuple<IndexT, WeightT>* vals_in =
      reinterpret_cast<WeightTuple<IndexT, WeightT>*>(work);
  WeightTuple<IndexT, WeightT>* vals_out = vals_in + nnz;

  void* nullwork = nullptr;
  size_t sort_bytes = 0;
  const int begin_bit = 0;
  const int end_bit = sizeof(IndexT) * 8;
  cub::DeviceRadixSort::SortPairs(nullwork,
                                  sort_bytes,
                                  cols,
                                  transpose_rows,
                                  vals_in,
                                  vals_out,
                                  nnz,
                                  begin_bit,
                                  end_bit,
                                  stream);

  size_t required_workspace = sort_bytes + buffer_bytes;

  if (work == nullptr) {
    *lwork = required_workspace;
    return;
  }

  assert(*lwork >= required_workspace);

  const int nthreads = DEFAULT_THREADS_PER_CTA;
  PackToTuple<IndexT, WeightT>
      <<<(nnz + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
          rows, weights, nnz, vals_in);

  void* sort_workspace = reinterpret_cast<void*>(vals_out + nnz);
  cub::DeviceRadixSort::SortPairs(static_cast<void*>(sort_workspace),
                                  *lwork,
                                  cols,
                                  transpose_rows,
                                  vals_in,
                                  vals_out,
                                  nnz,
                                  begin_bit,
                                  end_bit,
                                  stream);

  ExtractFromTuple<IndexT, WeightT>
      <<<(nnz + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
          vals_out, nnz, transpose_cols, transpose_weights);
}

/**
 * @brief Reorders indices from sample-id-first ordering as is needed during
 * forward to table-index-first ordering needed for backward. Output indices are
 * produced in coordinate (COO) format.
 *
 * @tparam IndexT Index datatype
 *
 * @param rows Pointer to the lookup indices.
 * @param cols Pointer to the offsets (CSR format) used during forward. Must be
 * nullptr when launching for fixed hotness.
 * @param weights Pointer to the weight array used during forward. If nullptr,
 * will not produce transposed weights.
 * @param nnz Number of nonzeros.
 * @param transpose_rows Pointer to the output transposed table indices.
 * @param transpose_cols Pointer to the output transposed sparse indices.
 * @param transpose_weights Pointer to the transposed weight array. If input
 * weights is nullptr, then will not produce transposed weights.
 * @param work Pointer to scratch workspace. Set to nullptr for workspace query.
 * @param lwork Pointer to size of scratch workspace.
 * @param stream Optional. The cudaStream to launch the kernel asynchronously.
 * If not specified, will launch the kernel on default stream.
 */
template <typename IndexT, typename WeightT>
void Transpose(const IndexT* rows,
               const IndexT* cols,
               const WeightT* weights,
               const int nnz,
               IndexT* transpose_rows,
               IndexT* transpose_cols,
               WeightT* transpose_weights,
               char* work,
               size_t* lwork,
               const cudaStream_t stream = 0) {
  if (weights == nullptr) {
    TransposeUnweighted<IndexT>(
        rows, cols, nnz, transpose_rows, transpose_cols, work, lwork, stream);
  } else {
    TransposeWeighted<IndexT, WeightT>(rows,
                                       cols,
                                       weights,
                                       nnz,
                                       transpose_rows,
                                       transpose_cols,
                                       transpose_weights,
                                       work,
                                       lwork,
                                       stream);
  }
}

struct FlagNonzero {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T lhs, const T rhs) {
    return (lhs == rhs) ? 0 : 1;
  }
};

/**
 * @brief The indices which are initially distributed between 0 and
 * num_categories values, are remapped to the range of 0 and num_unique, e.g.
 * indices =  [4, 4, 7, 8, 8, 8, 18] -> remapped_indices = [0, 0, 1, 2, 2, 2, 3]
 *
 * Requires a workspace query in which the function is called first with the
 * "work" parameter set to NULL. The required temporary memory size in bytes
 * will be returned in the lwork parameter by reference.
 *
 * @tparam IndexT Index datatype
 *
 * @param indices Pointer to the lookup indices, grouped by index.
 * @param nnz Length of the indices array.
 * @param remapped_indices Pointer to the remapped lookup indices (output)
 * @param work Temporary workspace
 * @param lwork Size of workspace in bytes (input/output)
 * @param stream Optional. The cudaStream to launch the kernel asynchronously.
 * If not specified, will launch the kernel on default stream.
 */
template <typename IndexT>
void ComputeCompressedGradIndices(const IndexT* indices,
                                  const int nnz,
                                  IndexT* remapped_indices,
                                  char* work,
                                  size_t* lwork,
                                  const cudaStream_t stream = 0) {
  void* nullwork = nullptr;
  size_t scan_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullwork, scan_storage_bytes, indices, remapped_indices, nnz, stream);
  size_t ad_storage_bytes = 0;
  cub::DeviceAdjacentDifference::SubtractLeftCopy(nullwork,
                                                  ad_storage_bytes,
                                                  remapped_indices,
                                                  remapped_indices,
                                                  nnz,
                                                  FlagNonzero(),
                                                  stream);
  size_t required_workspace = std::max(scan_storage_bytes, ad_storage_bytes);

  // Workspace query
  if (work == nullptr) {
    *lwork = required_workspace;
    return;
  }

  assert(*lwork >= required_workspace);

  cub::DeviceAdjacentDifference::SubtractLeftCopy(reinterpret_cast<void*>(work),
                                                  ad_storage_bytes,
                                                  indices,
                                                  remapped_indices,
                                                  nnz,
                                                  FlagNonzero(),
                                                  stream);

  cudaMemsetAsync(remapped_indices, 0, sizeof(IndexT), stream);

  cub::DeviceScan::InclusiveSum(reinterpret_cast<void*>(work),
                                *lwork,
                                remapped_indices,
                                remapped_indices,
                                nnz,
                                stream);
}

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_INDEX_TRANSFORMS_CUH_
