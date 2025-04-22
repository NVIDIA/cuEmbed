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

template <typename ElemT, typename IndexT, bool fp16_math>
class TestAgainstCpuRef
    : public ::testing::TestWithParam<utils::AllocationOptions> {
 protected:
  TestAgainstCpuRef() {
    options_ = GetParam();
    AllocateHost(options_, &u_a);
    AllocateDevice(options_, u_a, &d_a);
    cuembed::utils::RunForwardReference<ElemT, IndexT, int, fp16_math>(
        options_,
        u_a.embedding,
        u_a.indices,
        u_a.offsets,
        u_a.weights,
        &u_a.result);

    const int nnz = u_a.indices.size();
    cuembed::utils::RunTransposeReference<IndexT, int, ElemT>(
        options_,
        u_a.indices,
        u_a.offsets,
        u_a.weights,
        nnz,
        &u_a.transpose_indices,
        &u_a.transpose_remapped_indices,
        &u_a.transpose_sample_ids,
        &u_a.transpose_weights);

    cuembed::utils::RunBackwardReference<ElemT, IndexT, int>(
        options_,
        u_a.grad_y,
        u_a.transpose_indices,
        u_a.transpose_remapped_indices,
        u_a.transpose_sample_ids,
        u_a.transpose_weights,
        u_a.offsets,
        nnz,
        &u_a.grad_embedding,
        &u_a.inverse_mapping);
  }

  void RunTestForward() {
    cuembed::utils::RunForward<ElemT, IndexT, int, fp16_math>(options_,
                                                              d_a.embedding,
                                                              d_a.indices,
                                                              d_a.offsets,
                                                              d_a.weights,
                                                              &d_a.result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CheckResultForward();
  }

  void RunTestTranspose() {
    const int nnz = d_a.indices.size();
    cuembed::utils::RunTranspose<IndexT, int, ElemT>(
        options_,
        d_a.indices,
        d_a.offsets,
        d_a.weights,
        nnz,
        &d_a.transpose_indices,
        &d_a.transpose_remapped_indices,
        &d_a.transpose_sample_ids,
        &d_a.transpose_weights,
        &d_a.sample_ids,
        &d_a.transpose_workspace);
    CheckResultTranspose();
  }

  void RunTestBackward() {
    const int nnz = d_a.indices.size();
    cuembed::utils::RunTranspose<IndexT, int, ElemT>(
        options_,
        d_a.indices,
        d_a.offsets,
        d_a.weights,
        nnz,
        &d_a.transpose_indices,
        &d_a.transpose_remapped_indices,
        &d_a.transpose_sample_ids,
        &d_a.transpose_weights,
        &d_a.sample_ids,
        &d_a.transpose_workspace);

    const int num_unique = options_.compressed_grad()
                               ? d_a.transpose_remapped_indices.back() + 1
                               : 0;
    cuembed::utils::RunBackward<ElemT, IndexT, int>(
        options_,
        d_a.grad_y,
        d_a.transpose_indices,
        d_a.transpose_remapped_indices,
        d_a.transpose_sample_ids,
        d_a.transpose_weights,
        d_a.offsets,
        nnz,
        num_unique,
        &d_a.grad_embedding,
        &d_a.inverse_mapping);
    CheckResultBackward();
  }

 private:
  void CheckResultForward() {
    if (options_.combine_mode() == CombineMode::kSum) {
      EXPECT_EQ(d_a.result.size(),
                options_.batch_size() * options_.embed_width());
    } else if (options_.combine_mode() == CombineMode::kConcat) {
      EXPECT_EQ(
          d_a.result.size(),
          options_.batch_size() * options_.hotness() * options_.embed_width());
    } else if (options_.combine_mode() == CombineMode::kMean) {
      EXPECT_EQ(d_a.result.size(),
                options_.batch_size() * options_.embed_width());
    } else {
      EXPECT_TRUE(false) << "Reduce type not supported";
    }

    // Weighted summation on host vs. device may not be exact.
    if (options_.is_weighted()) {
      const float tolerance = 1e-4f;
      EXPECT_TRUE(thrust::equal(d_a.result.begin(),
                                d_a.result.end(),
                                u_a.result.begin(),
                                cuembed::utils::Near(tolerance)));
    } else {
      EXPECT_TRUE(thrust::equal(
          d_a.result.begin(), d_a.result.end(), u_a.result.begin()));
    }
  }

  void CheckResultTranspose() {
    EXPECT_TRUE(thrust::equal(d_a.transpose_indices.begin(),
                              d_a.transpose_indices.end(),
                              u_a.transpose_indices.begin()));
    EXPECT_TRUE(thrust::equal(d_a.transpose_remapped_indices.begin(),
                              d_a.transpose_remapped_indices.end(),
                              u_a.transpose_remapped_indices.begin()));

    // Check that sample_ids and weights sum to the same integer values
    // This allows transpose to order sample ids differently within an index
    IndexT d_sum = 0;
    IndexT ref_sum = 0;
    int64_t wt_sum = 0;
    int64_t ref_wt_sum = 0;
    for (int i = 0; i < u_a.transpose_sample_ids.size(); i++) {
      if (i > 0 && (u_a.transpose_indices[i - 1] != u_a.transpose_indices[i])) {
        EXPECT_TRUE(d_sum == ref_sum);
        d_sum = 0;
        ref_sum = 0;

        if (options_.is_weighted()) {
          EXPECT_TRUE(wt_sum == ref_wt_sum);
          wt_sum = 0;
          ref_wt_sum = 0;
        }
      }

      d_sum += d_a.transpose_sample_ids[i];
      ref_sum += u_a.transpose_sample_ids[i];

      if (options_.is_weighted()) {
        ElemT wt = d_a.transpose_weights[i];
        ElemT ref_wt = u_a.transpose_weights[i];
        int64_t wt_int = 0;
        int64_t ref_wt_int = 0;
        memcpy(&wt_int, &wt, std::min(sizeof(int64_t), sizeof(ElemT)));
        memcpy(&ref_wt_int, &ref_wt, std::min(sizeof(int64_t), sizeof(ElemT)));
        wt_sum += wt_int;
        ref_wt_sum += ref_wt_int;
      }
    }
  }

  void CheckResultBackward() {
    EXPECT_TRUE(thrust::equal(d_a.grad_embedding.begin(),
                              d_a.grad_embedding.end(),
                              u_a.grad_embedding.begin()));
    if (options_.compressed_grad()) {
      EXPECT_TRUE(thrust::equal(d_a.inverse_mapping.begin(),
                                d_a.inverse_mapping.end(),
                                u_a.inverse_mapping.begin()));
    }
  }

  utils::UniversalEmbeddingAllocation<ElemT, IndexT, int, ElemT, ElemT, ElemT>
      u_a;
  utils::DeviceEmbeddingAllocation<ElemT, IndexT, int, ElemT, ElemT, ElemT> d_a;
  utils::AllocationOptions options_;
};

// Some macros to make test allocation statement more concise.
#define ALLOC \
  (utils::AllocationOptions().num_categories(20_K).skip_grad_init(false))
#define SUM CombineMode::kSum
#define CONCAT CombineMode::kConcat
#define AVG CombineMode::kMean
#define CSR is_csr(true)
#define WEIGHTED is_weighted(true)
#define CMPGRAD compressed_grad(true)
#define BS batch_size
auto lookup_test_values = testing::Values(
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(SUM),
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(SUM).CSR,
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(AVG),
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(AVG).CSR,
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(CONCAT),
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(3).embed_width(2).hotness(4).combine_mode(CONCAT).CMPGRAD,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(SUM),
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(SUM).CSR,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(AVG),
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(AVG).CSR,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(CONCAT),
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(3).embed_width(4).hotness(4).combine_mode(CONCAT).CMPGRAD,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(SUM),
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(SUM).CSR,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(AVG),
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(AVG).CSR,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(CONCAT),
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(1023).embed_width(32).hotness(26).combine_mode(CONCAT).CMPGRAD,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(SUM),
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(SUM).CSR,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(AVG),
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(AVG).CSR,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(CONCAT),
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(1023).embed_width(36).hotness(26).combine_mode(CONCAT).CMPGRAD,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(SUM),
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(SUM).CSR,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(AVG),
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(AVG).CSR,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(CONCAT),
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(3).embed_width(512).hotness(63).combine_mode(CONCAT).CMPGRAD,
    ALLOC.BS(1023).embed_width(512).hotness(63).combine_mode(SUM),
    ALLOC.BS(1023).embed_width(512).hotness(63).combine_mode(SUM).CSR,
    ALLOC.BS(1023).embed_width(512).hotness(63).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(1023).embed_width(512).hotness(63).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(1023).embed_width(512).hotness(63).combine_mode(CONCAT),
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(SUM),
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(SUM).CSR,
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(SUM).WEIGHTED,
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(SUM).CSR.WEIGHTED,
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(CONCAT),
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(SUM).CMPGRAD,
    ALLOC.BS(1023).embed_width(514).hotness(63).combine_mode(CONCAT).CMPGRAD);
#undef ALLOC
#undef SUM
#undef CONCAT
#undef AVG
#undef CSR
#undef WEIGHTED
#undef CMPGRAD
#undef BS

class TestAgainstCpuEmbed32Idx32
    : public TestAgainstCpuRef<float, int32_t, false> {};
class TestAgainstCpuEmbed32Idx64
    : public TestAgainstCpuRef<float, int64_t, false> {};
class TestAgainstCpuEmbed16Idx32Reduce32
    : public TestAgainstCpuRef<__half, int32_t, true> {};
class TestAgainstCpuEmbed16Idx64Reduce32
    : public TestAgainstCpuRef<__half, int64_t, true> {};
class TestAgainstCpuEmbed16Idx32Reduce16
    : public TestAgainstCpuRef<__half, int32_t, false> {};
class TestAgainstCpuEmbed16Idx64Reduce16
    : public TestAgainstCpuRef<__half, int64_t, false> {};

std::string ToString(CombineMode value) {
  switch (value) {
    case CombineMode::kSum:
      return "Sum";
    case CombineMode::kConcat:
      return "Concat";
    case CombineMode::kMean:
      return "Mean";
    default:
      return "Unknown";
  }
}

std::string GenerateTestName(const utils::AllocationOptions& options) {
  std::string result = absl::StrFormat(
      "Width%dBatch%dHot%d%s%s%s%s",
      options.embed_width(),
      options.batch_size(),
      options.hotness(),
      ToString(options.combine_mode()),
      options.is_csr() ? "CSR" : "FixedHot",
      options.is_weighted() ? "Weight" : "NoWeight",
      options.compressed_grad() ? "CompressedGrad" : "FullGrad");
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed32Idx32,
    lookup_test_values,
    [](const testing::TestParamInfo<TestAgainstCpuEmbed32Idx32::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed32Idx64,
    lookup_test_values,
    [](const testing::TestParamInfo<TestAgainstCpuEmbed32Idx64::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed16Idx32Reduce16,
    lookup_test_values,
    [](const testing::TestParamInfo<
        TestAgainstCpuEmbed16Idx32Reduce16::ParamType>& info) {
      return GenerateTestName(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed16Idx64Reduce16,
    lookup_test_values,
    [](const testing::TestParamInfo<
        TestAgainstCpuEmbed16Idx64Reduce16::ParamType>& info) {
      return GenerateTestName(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed16Idx32Reduce32,
    lookup_test_values,
    [](const testing::TestParamInfo<
        TestAgainstCpuEmbed16Idx32Reduce32::ParamType>& info) {
      return GenerateTestName(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    EmbeddingLookup,
    TestAgainstCpuEmbed16Idx64Reduce32,
    lookup_test_values,
    [](const testing::TestParamInfo<
        TestAgainstCpuEmbed16Idx64Reduce32::ParamType>& info) {
      return GenerateTestName(info.param);
    });

TEST_P(TestAgainstCpuEmbed32Idx32, TestForward) { RunTestForward(); }
TEST_P(TestAgainstCpuEmbed32Idx64, TestForward) { RunTestForward(); }
TEST_P(TestAgainstCpuEmbed16Idx32Reduce16, TestForward) { RunTestForward(); }
TEST_P(TestAgainstCpuEmbed16Idx64Reduce16, TestForward) { RunTestForward(); }
TEST_P(TestAgainstCpuEmbed16Idx32Reduce32, TestForward) { RunTestForward(); }
TEST_P(TestAgainstCpuEmbed16Idx64Reduce32, TestForward) { RunTestForward(); }

TEST_P(TestAgainstCpuEmbed32Idx32, TestTranspose) { RunTestTranspose(); }
TEST_P(TestAgainstCpuEmbed32Idx64, TestTranspose) { RunTestTranspose(); }
TEST_P(TestAgainstCpuEmbed16Idx32Reduce16, TestTranspose) {
  RunTestTranspose();
}
TEST_P(TestAgainstCpuEmbed16Idx64Reduce16, TestTranspose) {
  RunTestTranspose();
}
TEST_P(TestAgainstCpuEmbed16Idx32Reduce32, TestTranspose) {
  RunTestTranspose();
}
TEST_P(TestAgainstCpuEmbed16Idx64Reduce32, TestTranspose) {
  RunTestTranspose();
}

TEST_P(TestAgainstCpuEmbed32Idx32, TestBackward) { RunTestBackward(); }
TEST_P(TestAgainstCpuEmbed32Idx64, TestBackward) { RunTestBackward(); }
TEST_P(TestAgainstCpuEmbed16Idx32Reduce16, TestBackward) { RunTestBackward(); }
TEST_P(TestAgainstCpuEmbed16Idx64Reduce16, TestBackward) { RunTestBackward(); }
TEST_P(TestAgainstCpuEmbed16Idx32Reduce32, TestBackward) { RunTestBackward(); }
TEST_P(TestAgainstCpuEmbed16Idx64Reduce32, TestBackward) { RunTestBackward(); }
}  // namespace cuembed
