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

#include "gtest/gtest.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace utils {

class EmbeddingAllocationTest : public ::testing::TestWithParam<CombineMode> {
 public:
  EmbeddingAllocationTest() {
    options_.num_categories(kNumCategories)
        .batch_size(kBatchSize)
        .hotness(kHotness)
        .alpha(kAlpha)
        .embed_width(kWidth);
  }
  void FinishSetup(const CombineMode combine_mode,
                   const bool is_csr,
                   const bool compressed_grad) {
    options_.combine_mode(combine_mode);
    options_.is_csr(is_csr);
    options_.compressed_grad(compressed_grad);
    AllocateHost(options_, &u_a);
  }

  void RunTest(const CombineMode combine_mode,
               const bool is_csr,
               const bool compressed_grad) {
    FinishSetup(combine_mode, is_csr, compressed_grad);
    ValidateOptions(combine_mode, is_csr, compressed_grad);
    ValidateAllocations(combine_mode, is_csr, compressed_grad);
    ValidateIndices(is_csr);
    ValidateWeights();
  }

 private:
  void ValidateOptions(const CombineMode combine_mode,
                       const bool is_csr,
                       const bool compressed_grad) {
    EXPECT_EQ(options_.num_categories(), kNumCategories);
    EXPECT_EQ(options_.batch_size(), kBatchSize);
    EXPECT_EQ(options_.alpha(), kAlpha);
    EXPECT_EQ(options_.embed_width(), kWidth);
    EXPECT_EQ(options_.combine_mode(), combine_mode);
    EXPECT_EQ(options_.is_csr(), is_csr);
    EXPECT_EQ(options_.compressed_grad(), compressed_grad);
  }

  void ValidateAllocations(const CombineMode combine_mode,
                           const bool is_csr,
                           const bool compressed_grad) {
    EXPECT_EQ(u_a.embedding.size(), kNumCategories * kWidth);
    if (compressed_grad) {
      EXPECT_LE(u_a.grad_embedding.size(), u_a.indices.size() * kWidth);
      EXPECT_LE(u_a.inverse_mapping.size(), u_a.indices.size());
    } else {
      EXPECT_EQ(u_a.grad_embedding.size(), kNumCategories * kWidth);
      EXPECT_EQ(u_a.inverse_mapping.size(), 0);
    }
    if (is_csr) {
      EXPECT_LE(u_a.indices.size(), kBatchSize * kHotness);
      EXPECT_EQ(u_a.indices.size(), u_a.offsets.back());
      EXPECT_EQ(u_a.transpose_remapped_indices.size(), u_a.offsets.back());
      EXPECT_EQ(u_a.transpose_sample_ids.size(), u_a.offsets.back());
      EXPECT_EQ(u_a.result.size(), kBatchSize * kWidth);
    } else {
      EXPECT_EQ(u_a.indices.size(), kBatchSize * kHotness);
      EXPECT_EQ(u_a.transpose_indices.size(), kBatchSize * kHotness);
      EXPECT_EQ(u_a.transpose_remapped_indices.size(), kBatchSize * kHotness);
      EXPECT_EQ(u_a.transpose_sample_ids.size(), kBatchSize * kHotness);
      if (combine_mode == CombineMode::kConcat) {
        EXPECT_EQ(u_a.result.size(), kBatchSize * kWidth * kHotness);
        EXPECT_EQ(u_a.grad_y.size(), kBatchSize * kWidth * kHotness);
      } else {
        EXPECT_EQ(u_a.result.size(), kBatchSize * kWidth);
        EXPECT_EQ(u_a.grad_y.size(), kBatchSize * kWidth);
      }
    }
  }

  void ValidateIndices(const bool is_csr) {
    for (int sample_id = 0; sample_id < kBatchSize; sample_id++) {
      const int hotness_for_sample =
          is_csr ? (u_a.offsets[sample_id + 1] - u_a.offsets[sample_id])
                 : options_.hotness();
      const int index_start =
          is_csr ? u_a.offsets[sample_id] : sample_id * options_.hotness();
      EXPECT_GE(hotness_for_sample, 0);

      // Check for no repetitions for a sample.
      std::set<int32_t> used_indices;
      for (int i_hotness = 0; i_hotness < hotness_for_sample; i_hotness++) {
        auto index = u_a.indices[index_start + i_hotness];
        EXPECT_TRUE(used_indices.find(index) == used_indices.end());
        used_indices.insert(index);
      }
    }
  }

  void ValidateWeights() {
    EXPECT_EQ(u_a.weights.size(), u_a.indices.size());
    for (auto weight : u_a.weights) {
      EXPECT_GE(weight, 0.0f);
      EXPECT_LE(weight, 1.0f);
    }
  }

  AllocationOptions options_;
  UniversalEmbeddingAllocation<float, int32_t, int, float, float, float> u_a;

  const int32_t kNumCategories = 7_M;
  const int32_t kBatchSize = 65_K;
  const int32_t kHotness = 32;
  const float kAlpha = 1.5f;
  const int32_t kWidth = 16;
};

TEST_P(EmbeddingAllocationTest, FixedHotnessWorks) {
  auto combine_mode = GetParam();
  const bool is_csr = false;
  const bool compressed_grad = false;
  RunTest(combine_mode, is_csr, compressed_grad);
}

TEST_P(EmbeddingAllocationTest, CSRWorks) {
  auto combine_mode = GetParam();
  if (combine_mode == CombineMode::kConcat) {
    return;
  }
  const bool is_csr = true;
  const bool compressed_grad = false;
  RunTest(combine_mode, is_csr, compressed_grad);
}

TEST_P(EmbeddingAllocationTest, SparseGradientWorks) {
  auto combine_mode = GetParam();
  const bool is_csr = false;
  const bool compressed_grad = false;
  RunTest(combine_mode, is_csr, compressed_grad);
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingAllocationTest,
    EmbeddingAllocationTest,
    ::testing::Values(
        CombineMode::kConcat,
        CombineMode::kSum)  // This runs the test for each of these values
);

}  // namespace utils
}  // namespace cuembed
