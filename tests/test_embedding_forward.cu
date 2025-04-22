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
#include "cuembed/include/embedding_lookup.cuh"
#include "gtest/gtest.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// CPU reference implementations
#include "utils/include/embedding_lookup_cpu.hpp"

namespace cuembed {

enum class DeviceType { kGPU, kCPU };

template <typename T>
class EmbeddingRefTest : public ::testing::Test {
 public:
  void LaunchTest(const CombineMode mode,
                  const DeviceType device,
                  const bool is_csr,
                  const bool is_weighted) {
    this->CreateResult(mode);
    this->LaunchKernel(mode, device, is_csr, is_weighted);
    this->CheckResult(mode, is_weighted);
  }

  typedef typename T::EmbedType EmbedType;
  typedef typename T::IndexType IndexType;
  typedef int OffsetType;

 private:
  void CreateResult(const CombineMode mode) {
    if (mode != CombineMode::kConcat) {
      result_.reset(
          new thrust::universal_vector<EmbedType>(batch_size_ * embed_width_));
    } else {
      result_.reset(new thrust::universal_vector<EmbedType>(
          batch_size_ * embed_width_ * hotness_));
    }
  }
  void LaunchKernel(const CombineMode mode,
                    const DeviceType device,
                    const bool is_csr,
                    const bool is_weighted) {
    int* offsets = is_csr ? offsets_.data().get() : nullptr;
    int hotness = is_csr ? 0 : hotness_;
    EmbedType* weights = (mode == CombineMode::kConcat || (!is_weighted))
                             ? nullptr
                             : weights_.data().get();
    const bool kFp16Math = false;
    if (device == DeviceType::kCPU) {
      EmbeddingForwardCpu<EmbedType,
                          EmbedType,
                          IndexType,
                          OffsetType,
                          kFp16Math>(embedding_.data().get(),
                                     embed_width_,
                                     batch_size_,
                                     hotness,
                                     indices_.data().get(),
                                     offsets,
                                     weights,
                                     result_->data().get(),
                                     mode);
    } else if (device == DeviceType::kGPU) {
      EmbeddingForward<EmbedType, EmbedType, IndexType, OffsetType, kFp16Math>(
          embedding_.data().get(),
          embed_width_,
          indices_.data().get(),
          offsets,
          weights,
          batch_size_,
          hotness,
          mode,
          result_->data().get());
      CHECK_CUDA(cudaDeviceSynchronize());
    } else {
      CHECK(false) << "not supported device";
    }
  }
  void CheckResult(const CombineMode mode, const bool is_weighted) {
    const thrust::universal_vector<EmbedType>* reference = nullptr;
    if (mode == CombineMode::kSum && is_weighted) {
      reference = &ref_result_sum_weighted_;
    } else if (mode == CombineMode::kSum && (!is_weighted)) {
      reference = &ref_result_sum_;
    } else if (mode == CombineMode::kConcat) {
      reference = &ref_result_concat_;
    } else if (mode == CombineMode::kMean) {
      reference = &ref_result_avg_;
    }
    for (size_t i = 0; i < result_->size(); ++i) {
      EXPECT_EQ(result_->data()[i], reference->data()[i]);
    }
  }

 private:
  const int embed_width_{4};
  const int hotness_{2};
  thrust::universal_vector<EmbedType> embedding_{
      1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10.,
      11., 12., 13., 14., 15., 16., 17., 18., 19., 20.};
  const int batch_size_{2};
  thrust::universal_vector<IndexType> indices_{1, 3, 0, 4};
  thrust::universal_vector<int> offsets_{0, 2, 4};
  thrust::universal_vector<EmbedType> weights_{1.f, 0.5f, 1.f, 0.5f};
  const thrust::universal_vector<EmbedType> ref_result_concat_{
      5., 6., 7., 8., 13., 14., 15., 16., 1., 2., 3., 4., 17., 18., 19., 20.};
  const thrust::universal_vector<EmbedType> ref_result_sum_{
      18.,
      20.,
      22.,
      24.,
      18.,
      20.,
      22.,
      24.,
  };
  const thrust::universal_vector<EmbedType> ref_result_avg_{
      9.,
      10.,
      11.,
      12.,
      9.,
      10.,
      11.,
      12.,
  };
  const thrust::universal_vector<EmbedType> ref_result_sum_weighted_{
      11.5,
      13.,
      14.5,
      16.,
      9.5,
      11.,
      12.5,
      14.,
  };

  std::unique_ptr<thrust::universal_vector<EmbedType>> result_;
};

TYPED_TEST_SUITE_P(EmbeddingRefTest);

TYPED_TEST_P(EmbeddingRefTest, TestFixedHotnessAgainstRefCpu) {
  const bool kIsCSR = false;
  for (const auto weighted : {true, false}) {
    for (const auto mode :
         {CombineMode::kSum, CombineMode::kConcat, CombineMode::kMean}) {
      if (mode == CombineMode::kMean && weighted) {
        continue;
      }
      this->LaunchTest(mode, DeviceType::kCPU, kIsCSR, weighted);
    }
  }
}

TYPED_TEST_P(EmbeddingRefTest, TestCSRAgainstRefCpu) {
  const bool kIsCSR = true;
  for (const auto weighted : {true, false}) {
    for (const auto mode : {CombineMode::kSum, CombineMode::kMean}) {
      if (mode == CombineMode::kMean && weighted) {
        continue;
      }
      this->LaunchTest(mode, DeviceType::kCPU, kIsCSR, weighted);
    }
  }
}

TYPED_TEST_P(EmbeddingRefTest, TestFixedHotnessAgainstRefGpu) {
  const bool kIsCSR = false;
  for (const auto weighted : {true, false}) {
    for (const auto mode :
         {CombineMode::kSum, CombineMode::kConcat, CombineMode::kMean}) {
      if (mode == CombineMode::kMean && weighted) {
        continue;
      }
      this->LaunchTest(mode, DeviceType::kGPU, kIsCSR, weighted);
    }
  }
}

TYPED_TEST_P(EmbeddingRefTest, TestCSRAgainstRefGpu) {
  const bool kIsCSR = true;
  for (const auto weighted : {true, false}) {
    for (const auto mode : {CombineMode::kSum, CombineMode::kMean}) {
      if (mode == CombineMode::kMean && weighted) {
        continue;
      }
      this->LaunchTest(mode, DeviceType::kGPU, kIsCSR, weighted);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingRefTest,
                            TestFixedHotnessAgainstRefCpu,
                            TestCSRAgainstRefCpu,
                            TestFixedHotnessAgainstRefGpu,
                            TestCSRAgainstRefGpu);

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

char ForwardRefTestNameGlobal[] = "ForwardRefTest_";
INSTANTIATE_TYPED_TEST_SUITE_P(
    Embedding,
    EmbeddingRefTest,
    EmbedTestTypes,
    utils::EmbeddingRefTestNames<ForwardRefTestNameGlobal>);

}  // namespace cuembed
