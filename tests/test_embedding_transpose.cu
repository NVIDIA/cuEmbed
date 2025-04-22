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

#include <cuda_fp16.h>
#include <thrust/universal_vector.h>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "cuembed/include/index_transforms.cuh"
#include "gtest/gtest.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// CPU reference implementations
#include "utils/include/index_transforms_cpu.hpp"

namespace cuembed {

enum class DeviceType { kGPU, kCPU };

template <typename T>
class TransposeRefTest : public ::testing::Test {
 public:
  void LaunchTest(const DeviceType device, const bool is_weighted) {
    this->CreateResult();
    this->LaunchKernel(device, is_weighted);
    this->CheckResult(is_weighted);
  }

  typedef typename T::EmbedType EmbedType;
  typedef typename T::IndexType IndexType;

 private:
  void CreateResult() {
    transpose_indices_.reset(new thrust::universal_vector<IndexType>(nnz_));
    transpose_sample_ids_.reset(new thrust::universal_vector<IndexType>(nnz_));
    transpose_weights_.reset(new thrust::universal_vector<EmbedType>(nnz_));
  }
  void LaunchKernel(const DeviceType device, const bool is_weighted) {
    EmbedType* weights = (!is_weighted) ? nullptr : weights_.data().get();
    EmbedType* transpose_weights =
        (!is_weighted) ? nullptr : transpose_weights_->data().get();
    if (device == DeviceType::kCPU) {
      TransposeCpu<IndexType, EmbedType>(sample_ids_.data().get(),
                                         indices_.data().get(),
                                         weights,
                                         nnz_,
                                         transpose_indices_->data().get(),
                                         transpose_sample_ids_->data().get(),
                                         transpose_weights);
    } else if (device == DeviceType::kGPU) {
      size_t lwork;
      Transpose<IndexType, EmbedType>(sample_ids_.data().get(),
                                      indices_.data().get(),
                                      weights,
                                      nnz_,
                                      transpose_indices_->data().get(),
                                      transpose_sample_ids_->data().get(),
                                      transpose_weights,
                                      nullptr,
                                      &lwork);

      thrust::device_vector<char> t_work(lwork);

      Transpose<IndexType, EmbedType>(sample_ids_.data().get(),
                                      indices_.data().get(),
                                      weights,
                                      nnz_,
                                      transpose_indices_->data().get(),
                                      transpose_sample_ids_->data().get(),
                                      transpose_weights,
                                      t_work.data().get(),
                                      &lwork);
      CHECK_CUDA(cudaDeviceSynchronize());
    } else {
      CHECK(false) << "not supported device";
    }
  }
  void CheckResult(const bool is_weighted) {
    for (int64_t i = 0; i < nnz_; ++i) {
      EXPECT_EQ(transpose_indices_->data()[i],
                ref_transpose_indices_.data()[i]);

      // No repeated indices so exact match is ensured for weights and
      // sample_ids
      EXPECT_EQ(transpose_sample_ids_->data()[i],
                ref_transpose_sample_ids_.data()[i]);
      if (is_weighted) {
        EXPECT_EQ(transpose_weights_->data()[i],
                  ref_transpose_weights_.data()[i]);
      }
    }
  }

 private:
  const int nnz_{4};
  thrust::universal_vector<IndexType> indices_{1, 3, 0, 4};
  thrust::universal_vector<IndexType> sample_ids_{0, 0, 1, 1};
  thrust::universal_vector<EmbedType> weights_{1.f, 0.5f, 1.f, 0.5f};
  const thrust::universal_vector<IndexType> ref_transpose_indices_{0, 1, 3, 4};
  const thrust::universal_vector<IndexType> ref_transpose_sample_ids_{
      1, 0, 0, 1};
  const thrust::universal_vector<IndexType> ref_transpose_sample_ids_concat_{
      2, 0, 1, 3};
  const thrust::universal_vector<EmbedType> ref_transpose_weights_{
      1.f, 1.f, 0.5f, 0.5f};

  std::unique_ptr<thrust::universal_vector<IndexType>> transpose_indices_;
  std::unique_ptr<thrust::universal_vector<IndexType>> transpose_sample_ids_;
  std::unique_ptr<thrust::universal_vector<EmbedType>> transpose_weights_;
};

TYPED_TEST_SUITE_P(TransposeRefTest);

TYPED_TEST_P(TransposeRefTest, TestAgainstRefCpu) {
  for (const auto weighted : {true, false}) {
    this->LaunchTest(DeviceType::kCPU, weighted);
  }
}

TYPED_TEST_P(TransposeRefTest, TestAgainstRefGpu) {
  for (const auto weighted : {true, false}) {
    this->LaunchTest(DeviceType::kGPU, weighted);
  }
}

REGISTER_TYPED_TEST_SUITE_P(TransposeRefTest,
                            TestAgainstRefCpu,
                            TestAgainstRefGpu);

template <typename EmbedT, typename IndexT>
struct EmbedTestTypeCombo {
  typedef EmbedT EmbedType;
  typedef IndexT IndexType;
};

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>,
                         EmbedTestTypeCombo<float, int64_t>,
                         EmbedTestTypeCombo<__half, int32_t>,
                         EmbedTestTypeCombo<__half, int64_t>>
    EmbedTestTypes;

char TransposeRefTestNameGlobal[] = "TransposeRefTest_";
INSTANTIATE_TYPED_TEST_SUITE_P(
    Transpose,
    TransposeRefTest,
    EmbedTestTypes,
    utils::EmbeddingRefTestNames<TransposeRefTestNameGlobal>);

}  // namespace cuembed
