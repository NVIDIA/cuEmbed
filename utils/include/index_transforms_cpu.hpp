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

#ifndef UTILS_INCLUDE_INDEX_TRANSFORMS_CPU_HPP_
#define UTILS_INCLUDE_INDEX_TRANSFORMS_CPU_HPP_

#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cuembed/include/embedding_lookup_types.cuh"

namespace cuembed {

template <typename IndexT>
void ExtractRowIdsFromFixedCpu(const int batch_size,
                               const int num_hots,
                               IndexT* row_ids) {
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_hots; h++) {
      row_ids[b * num_hots + h] = static_cast<IndexT>(b);
    }
  }
}

template <typename IndexT, typename OffsetT>
void ExtractRowIdsFromCSRCpu(const OffsetT* offsets,
                             const int batch_size,
                             IndexT* row_ids) {
  int cnt = 0;
  for (int b = 0; b < batch_size; b++) {
    for (OffsetT o = offsets[b]; o < offsets[b + 1]; o++) {
      row_ids[cnt] = static_cast<IndexT>(b);
      cnt++;
    }
  }
}

template <typename IndexT>
void ExtractRowIdsForConcatCpu(const int nnz, IndexT* row_ids) {
  for (int i = 0; i < nnz; i++) {
    row_ids[i] = static_cast<IndexT>(i);
  }
}

template <typename IndexT>
void ComputeCompressedGradIndicesCpu(const IndexT* indices,
                                     const int nnz,
                                     IndexT* remapped_indices) {
  IndexT unique_cnt = 0;
  for (int64_t cnt = 0; cnt < nnz; cnt++) {
    if ((cnt > 0) && (indices[cnt] != indices[cnt - 1])) {
      unique_cnt++;
    }
    remapped_indices[cnt] = unique_cnt;
  }
}

template <typename IndexT, typename WeightT>
struct index_tuple {
  IndexT idx;
  IndexT sid;
  WeightT wt;
};

template <typename IndexT, typename WeightT>
void TransposeCpu(const IndexT* rows,
                  const IndexT* cols,
                  const WeightT* weights,
                  const int nnz,
                  IndexT* transpose_rows,
                  IndexT* transpose_cols,
                  WeightT* transpose_weights) {
  // Fill indices and weights into vector
  std::vector<index_tuple<IndexT, WeightT> > tuples;
  for (int cnt = 0; cnt < nnz; cnt++) {
    index_tuple<IndexT, WeightT> tuple;
    tuple.idx = cols[cnt];
    tuple.sid = rows[cnt];
    tuple.wt = (weights != nullptr) ? weights[cnt] : static_cast<WeightT>(0);
    tuples.push_back(tuple);
  }

  // Sort (offsets, indices, weights)
  std::sort(tuples.begin(),
            tuples.end(),
            [](const index_tuple<IndexT, WeightT>& a,
               const index_tuple<IndexT, WeightT>& b) {
              if (a.idx < b.idx) return true;
              if (a.idx > b.idx) return false;
              if (a.sid < b.sid) return true;
              if (a.sid > b.sid) return false;
              if (a.wt < b.wt) return true;
              return false;
            });

  // Copy to output
  for (int64_t cnt = 0; cnt < nnz; cnt++) {
    transpose_rows[cnt] = tuples[cnt].idx;
    transpose_cols[cnt] = tuples[cnt].sid;
    if (transpose_weights != nullptr) {
      transpose_weights[cnt] = tuples[cnt].wt;
    }
  }
}

}  // namespace cuembed

#endif  // UTILS_INCLUDE_INDEX_TRANSFORMS_CPU_HPP_
