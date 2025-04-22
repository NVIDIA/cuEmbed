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
#include "cuembed/include/index_transforms.cuh"
#include "gtest/gtest.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// CPU reference implementations
#include "utils/include/embedding_lookup_cpu.hpp"
#include "utils/include/index_transforms_cpu.hpp"

namespace cuembed {

enum class DeviceType { kGPU, kCPU };

template <typename T>
class EmbeddingBackwardRefTest : public ::testing::Test {
 public:
  void LaunchTest(const CombineMode mode,
                  const DeviceType device,
                  const bool is_weighted,
                  const bool compressed_grad,
                  const bool skip_grad_init = false) {
    this->CreateResult(compressed_grad, skip_grad_init);
    this->LaunchKernel(
        mode, device, is_weighted, compressed_grad, skip_grad_init);
    this->CheckResult(mode, is_weighted, compressed_grad, skip_grad_init);
  }

  typedef typename T::EmbedType EmbedType;
  typedef typename T::IndexType IndexType;
  typedef int OffsetType;

 private:
  void CreateResult(const bool compressed_grad, const bool skip_grad_init) {
    grad_embedding_.reset(new thrust::universal_vector<EmbedType>(
        num_categories_ * embed_width_));
    // If kernel doesn't initialize the gradient then do it here
    if (skip_grad_init) {
      thrust::fill(grad_embedding_->begin(), grad_embedding_->end(), 0);
    }
    if (compressed_grad) {
      inverse_mapping_.reset(
          new thrust::universal_vector<IndexType>(num_categories_));
    }
  }
  void LaunchKernel(const CombineMode mode,
                    const DeviceType device,
                    const bool is_weighted,
                    const bool compressed_grad,
                    const bool skip_grad_init) {
    const EmbedType* transpose_weights =
        (!is_weighted) ? nullptr : transpose_weights_.data().get();
    const thrust::universal_vector<EmbedType>* grad_y = nullptr;
    const thrust::universal_vector<IndexType>* transpose_sample_ids = nullptr;
    if (mode == CombineMode::kSum || mode == CombineMode::kMean) {
      grad_y = &grad_y_sum_;
      transpose_sample_ids = &transpose_sample_ids_;
    } else if (mode == CombineMode::kConcat) {
      grad_y = &grad_y_concat_;
      transpose_sample_ids = &transpose_sample_ids_concat_;
    }
    const IndexType* transpose_remapped_indices =
        (compressed_grad) ? transpose_remapped_indices_.data().get() : nullptr;
    const int num_grad_embedding_rows =
        (compressed_grad) ? num_unique_ : num_categories_;
    IndexType* inverse_mapping =
        (compressed_grad) ? inverse_mapping_->data().get() : nullptr;
    if (device == DeviceType::kCPU) {
      EmbeddingBackwardCpu<EmbedType, IndexType>(
          grad_y->data().get(),
          embed_width_,
          num_grad_embedding_rows,
          nnz_,
          transpose_indices_.data().get(),
          transpose_sample_ids->data().get(),
          transpose_remapped_indices,
          transpose_weights,
          skip_grad_init,
          grad_embedding_->data().get(),
          inverse_mapping);
    } else if (device == DeviceType::kGPU) {
      EmbeddingBackward<EmbedType, IndexType>(
          grad_y->data().get(),
          embed_width_,
          num_grad_embedding_rows,
          nnz_,
          transpose_indices_.data().get(),
          transpose_sample_ids->data().get(),
          transpose_remapped_indices,
          transpose_weights,
          skip_grad_init,
          grad_embedding_->data().get(),
          inverse_mapping);
      CHECK_CUDA(cudaDeviceSynchronize());
    } else {
      CHECK(false) << "not supported device";
    }
  }
  void CheckResult(const CombineMode mode,
                   const bool is_weighted,
                   const bool compressed_grad,
                   const bool skip_grad_init) {
    const thrust::universal_vector<EmbedType>* reference = nullptr;
    if (compressed_grad) {
      if (mode == CombineMode::kSum && is_weighted) {
        reference = &ref_compressed_grad_embedding_sum_weighted_;
      } else if (mode == CombineMode::kSum && (!is_weighted)) {
        reference = &ref_compressed_grad_embedding_sum_;
      } else if (mode == CombineMode::kConcat && (!is_weighted)) {
        reference = &ref_compressed_grad_embedding_concat_;
      } else if (mode == CombineMode::kConcat && is_weighted) {
        reference = &ref_compressed_grad_embedding_concat_weighted_;
      }
    } else {
      if (mode == CombineMode::kSum && is_weighted) {
        reference = &ref_grad_embedding_sum_weighted_;
      } else if (mode == CombineMode::kSum && (!is_weighted)) {
        reference = &ref_grad_embedding_sum_;
      } else if (mode == CombineMode::kConcat && (!is_weighted)) {
        reference = &ref_grad_embedding_concat_;
      } else if (mode == CombineMode::kConcat && is_weighted) {
        reference = &ref_grad_embedding_concat_weighted_;
      }
    }
    if (reference != nullptr) {
      for (size_t i = 0; i < grad_embedding_->size(); ++i) {
        EXPECT_EQ(grad_embedding_->data()[i], reference->data()[i]);
      }
    }
    if (compressed_grad) {
      for (size_t i = 0; i < num_unique_; ++i) {
        EXPECT_EQ(inverse_mapping_->data()[i], ref_inverse_mapping_.data()[i]);
      }
    }
  }

 private:
  const int embed_width_{4};
  const int hotness_{2};
  const int num_categories_{5};
  const int batch_size_{2};
  const int nnz_{4};
  const int num_unique_{3};
  const thrust::universal_vector<IndexType> transpose_indices_{0, 1, 3, 3};
  const thrust::universal_vector<IndexType> transpose_remapped_indices_{
      0, 1, 2, 2};
  const thrust::universal_vector<IndexType> transpose_sample_ids_{1, 0, 0, 1};
  const thrust::universal_vector<IndexType> transpose_sample_ids_concat_{
      2, 0, 1, 3};
  const thrust::universal_vector<EmbedType> transpose_weights_{
      3.f, 1.f, 0.5f, 3.f};
  const thrust::universal_vector<EmbedType> grad_y_sum_{
      1., 2., 3., 4., 5., 6., 7., 8.};
  const thrust::universal_vector<EmbedType> grad_y_concat_{
      1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.};
  thrust::universal_vector<EmbedType> ref_grad_embedding_sum_{
      5., 6., 7., 8., 1.,  2.,  3., 4., 0., 0.,
      0., 0., 6., 8., 10., 12., 0., 0., 0., 0.};
  thrust::universal_vector<EmbedType> ref_grad_embedding_sum_weighted_{
      15., 18., 21.,  24., 1.,   2.,  3., 4., 0., 0.,
      0.,  0.,  15.5, 19., 22.5, 26., 0., 0., 0., 0.};
  thrust::universal_vector<EmbedType> ref_grad_embedding_concat_{
      9., 10., 11., 12., 1.,  2.,  3., 4., 0., 0.,
      0., 0.,  18., 20., 22., 24., 0., 0., 0., 0.};
  thrust::universal_vector<EmbedType> ref_grad_embedding_concat_weighted_{
      27., 30., 33.,  36., 1.,   2.,  3., 4., 0., 0.,
      0.,  0.,  41.5, 45., 48.5, 52., 0., 0., 0., 0.};
  thrust::universal_vector<IndexType> ref_inverse_mapping_{0, 1, 3};
  thrust::universal_vector<EmbedType> ref_compressed_grad_embedding_sum_{
      5., 6., 7., 8., 1., 2., 3., 4., 6., 8., 10., 12.};
  thrust::universal_vector<EmbedType>
      ref_compressed_grad_embedding_sum_weighted_{
          15., 18., 21., 24., 1., 2., 3., 4., 15.5, 19., 22.5, 26.};
  thrust::universal_vector<EmbedType> ref_compressed_grad_embedding_concat_{
      9., 10., 11., 12., 1., 2., 3., 4., 18., 20., 22., 24.};
  thrust::universal_vector<EmbedType>
      ref_compressed_grad_embedding_concat_weighted_{
          27., 30., 33., 36., 1., 2., 3., 4., 41.5, 45., 48.5, 52.};

  std::unique_ptr<thrust::universal_vector<EmbedType>> grad_embedding_;
  std::unique_ptr<thrust::universal_vector<IndexType>> inverse_mapping_;
};

TYPED_TEST_SUITE_P(EmbeddingBackwardRefTest);

TYPED_TEST_P(EmbeddingBackwardRefTest, TestFixedHotnessAgainstRefCpu) {
  for (const auto compressed_grad : {false}) {
    for (const auto weighted : {false, true}) {
      for (const auto mode : {CombineMode::kSum, CombineMode::kConcat}) {
        this->LaunchTest(mode, DeviceType::kCPU, weighted, compressed_grad);
      }
    }
  }
}

TYPED_TEST_P(EmbeddingBackwardRefTest, TestFixedHotnessAgainstRefGpu) {
  for (const auto compressed_grad : {false}) {
    for (const auto weighted : {false, true}) {
      for (const auto mode : {CombineMode::kSum, CombineMode::kConcat}) {
        this->LaunchTest(mode, DeviceType::kGPU, weighted, compressed_grad);
      }
    }
  }
}

TYPED_TEST_P(EmbeddingBackwardRefTest, TestCSRAgainstRefCpu) {
  for (const auto compressed_grad : {false}) {
    for (const auto weighted : {false, true}) {
      for (const auto mode : {CombineMode::kSum, CombineMode::kConcat}) {
        this->LaunchTest(mode, DeviceType::kCPU, weighted, compressed_grad);
      }
    }
  }
}

TYPED_TEST_P(EmbeddingBackwardRefTest, TestCSRAgainstRefGpu) {
  for (const auto compressed_grad : {false}) {
    for (const auto weighted : {false, true}) {
      for (const auto mode : {CombineMode::kSum, CombineMode::kConcat}) {
        this->LaunchTest(mode, DeviceType::kGPU, weighted, compressed_grad);
      }
    }
  }
}

TYPED_TEST_P(EmbeddingBackwardRefTest, TestSkipInitGradAgainstRefCpu) {
  const bool weighted = false;
  const auto mode = CombineMode::kSum;
  for (const auto skip_grad_init : {false, true}) {
    for (const auto compressed_grad : {false, true}) {
      this->LaunchTest(
          mode, DeviceType::kCPU, weighted, compressed_grad, skip_grad_init);
    }
  }
}

TYPED_TEST_P(EmbeddingBackwardRefTest, TestSkipInitGradAgainstRefGpu) {
  const bool weighted = false;
  const auto mode = CombineMode::kSum;
  for (const auto skip_grad_init : {false, true}) {
    for (const auto compressed_grad : {false, true}) {
      this->LaunchTest(
          mode, DeviceType::kGPU, weighted, compressed_grad, skip_grad_init);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingBackwardRefTest,
                            TestFixedHotnessAgainstRefCpu,
                            TestFixedHotnessAgainstRefGpu,
                            TestCSRAgainstRefCpu,
                            TestCSRAgainstRefGpu,
                            TestSkipInitGradAgainstRefCpu,
                            TestSkipInitGradAgainstRefGpu);

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

char EmbeddingBackwardRefTestNameGlobal[] = "EmbeddingBackwardRefTest_";
INSTANTIATE_TYPED_TEST_SUITE_P(
    EmbeddingBackward,
    EmbeddingBackwardRefTest,
    EmbedTestTypes,
    utils::EmbeddingRefTestNames<EmbeddingBackwardRefTestNameGlobal>);

}  // namespace cuembed
