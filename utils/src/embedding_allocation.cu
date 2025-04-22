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

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "absl/log/check.h"
#include "cuembed/include/index_transforms.cuh"
#include "utils/include/datagen.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace utils {
AllocationOptions& AllocationOptions::num_categories(int32_t num_categories) {
  num_categories_ = num_categories;
  return *this;
}

AllocationOptions& AllocationOptions::batch_size(int32_t batch_size) {
  batch_size_ = batch_size;
  return *this;
}

AllocationOptions& AllocationOptions::hotness(int32_t hotness) {
  hotness_ = hotness;
  return *this;
}

AllocationOptions& AllocationOptions::alpha(float alpha) {
  alpha_ = alpha;
  return *this;
}

AllocationOptions& AllocationOptions::combine_mode(CombineMode type) {
  combine_mode_ = type;
  return *this;
}

AllocationOptions& AllocationOptions::embed_width(int32_t embed_width) {
  embed_width_ = embed_width;
  return *this;
}

AllocationOptions& AllocationOptions::permute_indices(bool permute_indices) {
  permute_indices_ = permute_indices;
  return *this;
}

AllocationOptions& AllocationOptions::shuffle_indices(int32_t shuffle_indices) {
  shuffle_indices_ = shuffle_indices;
  return *this;
}

AllocationOptions& AllocationOptions::is_csr(bool is_csr) {
  is_csr_ = is_csr;
  return *this;
}

AllocationOptions& AllocationOptions::is_weighted(bool is_weighted) {
  is_weighted_ = is_weighted;
  return *this;
}

AllocationOptions& AllocationOptions::compressed_grad(bool compressed_grad) {
  compressed_grad_ = compressed_grad;
  return *this;
}

AllocationOptions& AllocationOptions::skip_grad_init(bool skip_grad_init) {
  skip_grad_init_ = skip_grad_init;
  return *this;
}

template <typename InputT,
          typename OutputT,
          typename IndexT,
          typename WeightT,
          typename OffsetT>
void AllocateForward(const AllocationOptions& options,
                     thrust::universal_vector<InputT>* embedding,
                     thrust::universal_vector<IndexT>* indices,
                     thrust::universal_vector<OffsetT>* offsets,
                     thrust::universal_vector<WeightT>* weights,
                     thrust::universal_vector<OutputT>* result) {
  CHECK(options.num_categories() > 0 && options.batch_size() > 0 &&
        options.hotness() > 0 && options.embed_width() > 0);
  embedding->resize(((int64_t)options.num_categories()) * options.embed_width(),
                    0);

  // Fill embedding with random values.
  std::default_random_engine rng(123456);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::generate(
      embedding->begin(), embedding->end(), [&] { return (InputT)dist(rng); });

  if (options.combine_mode() != CombineMode::kConcat) {
    result->resize(((int64_t)options.batch_size()) * options.embed_width(), 0);
  } else {
    result->resize(((int64_t)options.batch_size()) * options.embed_width() *
                       options.hotness(),
                   0);
  }

  // Generate offsets for CSR representation. Each batch may lookup a random
  // number of values (maximum num_hotness).
  offsets->resize(options.batch_size() + 1);
  CHECK_GT(options.batch_size(), 0);
  (*offsets)[0] = 0;  // Starting value
  {
    std::uniform_int_distribution<> distrib(0, options.hotness());
    for (int i = 0; i < options.batch_size(); ++i) {
      (*offsets)[i + 1] = (*offsets)[i] + distrib(rng);
    }
  }
  // Generate lookup indices. Generate num_hotness of indices for each sample.
  // Copy the first hotness_for_sample indices into generated indices.
  auto generator = index_generators::PowerLawFeatureGenerator<IndexT>(
      options.num_categories() - 1,
      options.hotness(),
      options.alpha(),
      options.shuffle_indices(),
      options.permute_indices(),
      index_generators::PowerLawType::kPsx);

  indices->clear();
  for (int i = 0; i < options.batch_size(); i++) {
    auto generated_idx = generator.getCategoryIndices();
    CHECK_EQ(generated_idx.size(), options.hotness());
    int hotness_for_sample = options.hotness();
    if (options.is_csr()) {
      hotness_for_sample = (*offsets)[i + 1] - (*offsets)[i];
    }
    indices->insert(indices->end(),
                    generated_idx.begin(),
                    generated_idx.begin() + hotness_for_sample);
  }

  // Generate weights. Weights are either 0.5f or 0.25f for easier correctness
  // checking.
  {
    weights->resize(indices->size());
    std::bernoulli_distribution distrib(0.5);
    for (size_t i = 0; i < weights->size(); i++) {
      (*weights)[i] = distrib(rng) ? 0.5f : 0.25f;
    }
  }
}

template <typename IndexT, typename WeightT>
void AllocateTranspose(
    const AllocationOptions& options,
    const int nnz,
    thrust::universal_vector<IndexT>* transpose_indices,
    thrust::universal_vector<IndexT>* transpose_remapped_indices,
    thrust::universal_vector<IndexT>* transpose_sample_ids,
    thrust::universal_vector<WeightT>* transpose_weights,
    thrust::universal_vector<IndexT>* sample_ids,
    thrust::universal_vector<char>* transpose_workspace) {
  transpose_indices->resize(nnz, 0);
  transpose_remapped_indices->resize(nnz, 0);
  transpose_sample_ids->resize(nnz, 0);
  transpose_weights->resize(nnz, 0);
  sample_ids->resize(nnz, 0);

  // Allocate scratch space
  size_t lwork_transpose = 0;
  size_t lwork_compressed_grad = 0;
  thrust::universal_vector<IndexT> tmp_sample_ids((int64_t)nnz);
  thrust::universal_vector<IndexT> tmp_indices((int64_t)nnz);
  thrust::universal_vector<WeightT> tmp_weights((int64_t)nnz);
  const WeightT* weight_ptr = nullptr;
  WeightT* transpose_weight_ptr = nullptr;
  if (options.is_weighted()) {
    weight_ptr = tmp_weights.data().get();
    transpose_weight_ptr = transpose_weights->data().get();
  }
  Transpose<IndexT, WeightT>(tmp_sample_ids.data().get(),
                             tmp_indices.data().get(),
                             weight_ptr,
                             nnz,
                             transpose_indices->data().get(),
                             transpose_sample_ids->data().get(),
                             transpose_weight_ptr,
                             nullptr,
                             &lwork_transpose);

  if (options.compressed_grad()) {
    ComputeCompressedGradIndices<IndexT>(
        transpose_indices->data().get(),
        nnz,
        transpose_remapped_indices->data().get(),
        nullptr,
        &lwork_compressed_grad);
  }

  transpose_workspace->resize(std::max(lwork_transpose, lwork_compressed_grad));
}

template <typename GradT, typename IndexT>
void AllocateBackward(const AllocationOptions& options,
                      thrust::universal_vector<GradT>* grad_y,
                      thrust::universal_vector<GradT>* grad_embedding,
                      thrust::universal_vector<IndexT>* inverse_mapping,
                      const int num_unique) {
  if (options.combine_mode() != CombineMode::kConcat) {
    grad_y->resize(((int64_t)options.batch_size()) * options.embed_width(), 0);
  } else {
    grad_y->resize(((int64_t)options.batch_size()) * options.embed_width() *
                       options.hotness(),
                   0);
  }
  std::default_random_engine rng(654321);
  std::uniform_int_distribution<int> dist(-10, 10);
  std::generate(
      grad_y->begin(), grad_y->end(), [&] { return (GradT)dist(rng); });

  if (options.compressed_grad()) {
    grad_embedding->resize((int64_t)num_unique * options.embed_width(), 0);
    inverse_mapping->resize(num_unique, 0);
  } else {
    grad_embedding->resize(
        ((int64_t)options.num_categories()) * options.embed_width(), 0);
    inverse_mapping->resize(0, 0);
  }
}

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
                  bool forward_only) {
  AllocateForward<InputT, OutputT, IndexT, WeightT, OffsetT>(
      options,
      &allocation->embedding,
      &allocation->indices,
      &allocation->offsets,
      &allocation->weights,
      &allocation->result);

  if (forward_only) {
    return;
  }

  int nnz = allocation->indices.size();
  AllocateTranspose<IndexT, WeightT>(options,
                                     nnz,
                                     &allocation->transpose_indices,
                                     &allocation->transpose_remapped_indices,
                                     &allocation->transpose_sample_ids,
                                     &allocation->transpose_weights,
                                     &allocation->sample_ids,
                                     &allocation->transpose_workspace);

  // Compute num_unique
  thrust::device_vector<IndexT> indices_copy = allocation->indices;

  // Sort the copy of the indices
  thrust::sort(indices_copy.begin(), indices_copy.end());

  // Get the end of the unique range
  auto unique_end = thrust::unique(indices_copy.begin(), indices_copy.end());

  // Calculate the number of unique indices
  int num_unique = unique_end - indices_copy.begin();

  AllocateBackward<GradT, IndexT>(options,
                                  &allocation->grad_y,
                                  &allocation->grad_embedding,
                                  &allocation->inverse_mapping,
                                  num_unique);
}

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
                                       GradT>& u_a,
    DeviceEmbeddingAllocation<InputT, IndexT, OffsetT, WeightT, OutputT, GradT>*
        d_a,
    bool forward_only) {
  // Resize the device vectors to match the universal vectors
  d_a->embedding.resize(u_a.embedding.size());
  d_a->indices.resize(u_a.indices.size());
  d_a->offsets.resize(u_a.offsets.size());
  d_a->weights.resize(u_a.weights.size());
  d_a->result.resize(u_a.result.size());
  if (!forward_only) {
    d_a->transpose_indices.resize(u_a.transpose_indices.size());
    d_a->transpose_remapped_indices.resize(
        u_a.transpose_remapped_indices.size());
    d_a->transpose_sample_ids.resize(u_a.transpose_sample_ids.size());
    d_a->transpose_weights.resize(u_a.transpose_weights.size());
    d_a->sample_ids.resize(u_a.sample_ids.size());
    d_a->transpose_workspace.resize(u_a.transpose_workspace.size());
    d_a->grad_y.resize(u_a.grad_y.size());
    d_a->grad_embedding.resize(u_a.grad_embedding.size());
    d_a->inverse_mapping.resize(u_a.inverse_mapping.size());
  }

  // Copy input data from universal vectors to device vectors
  thrust::copy(
      u_a.embedding.begin(), u_a.embedding.end(), d_a->embedding.begin());
  thrust::copy(u_a.indices.begin(), u_a.indices.end(), d_a->indices.begin());
  thrust::copy(u_a.offsets.begin(), u_a.offsets.end(), d_a->offsets.begin());
  thrust::copy(u_a.weights.begin(), u_a.weights.end(), d_a->weights.begin());
  if (!forward_only) {
    thrust::copy(u_a.grad_y.begin(), u_a.grad_y.end(), d_a->grad_y.begin());
  }
}

#define ALLOCATE_TEMPLATE(InputT, OutputT, IndexT, WeightT, OffsetT, GradT) \
  template void                                                             \
  AllocateHost<InputT, IndexT, OffsetT, WeightT, OutputT, GradT>(           \
      const AllocationOptions& options,                                     \
      UniversalEmbeddingAllocation<InputT,                                  \
                                   IndexT,                                  \
                                   OffsetT,                                 \
                                   WeightT,                                 \
                                   OutputT,                                 \
                                   GradT>* u_a,                             \
      bool forward_only);                                                   \
  template void                                                             \
  AllocateDevice<InputT, IndexT, OffsetT, WeightT, OutputT, GradT>(         \
      const AllocationOptions& options,                                     \
      const UniversalEmbeddingAllocation<InputT,                            \
                                         IndexT,                            \
                                         OffsetT,                           \
                                         WeightT,                           \
                                         OutputT,                           \
                                         GradT>& u_a,                       \
      DeviceEmbeddingAllocation<InputT,                                     \
                                IndexT,                                     \
                                OffsetT,                                    \
                                WeightT,                                    \
                                OutputT,                                    \
                                GradT>* d_a,                                \
      bool forward_only);

ALLOCATE_TEMPLATE(float, float, int32_t, float, int, float)
ALLOCATE_TEMPLATE(float, float, int64_t, float, int, float)
ALLOCATE_TEMPLATE(__half, __half, int32_t, __half, int, __half)
ALLOCATE_TEMPLATE(__half, __half, int64_t, __half, int, __half)
ALLOCATE_TEMPLATE(__half, __half, int32_t, __half, int, float)
ALLOCATE_TEMPLATE(__half, __half, int64_t, __half, int, float)

}  // namespace utils

}  // namespace cuembed
