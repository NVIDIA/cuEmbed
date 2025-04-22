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

#include <random>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "cuembed/include/embedding_lookup_ops.cuh"
#include "gtest/gtest.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

namespace cuembed {

namespace {
bool Equal(const float4& a, const float4& b) {
  return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w));
}

bool Equal(const half4& a, const half4& b) {
  return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w));
}

bool Equal(const float2& a, const float2& b) {
  return ((a.x == b.x) && (a.y == b.y));
}

bool Equal(const half2& a, const half2& b) {
  return ((a.x == b.x) && (a.y == b.y));
}

template <typename VecT>
VecT GetZeroVec() {
  VecT result;
  memset(&result, 0, sizeof(VecT));
  return result;
}
}  // namespace

template <typename ElemT,
          typename LoadVecT,
          typename ReduceVecT,
          typename IndexT,
          typename OffsetT>
class OperationsTest
    : public ::testing::TestWithParam<utils::AllocationOptions> {
 protected:
  OperationsTest() {
    options_ = GetParam();
    AllocateHost(options_, &u_a);

    nnz_ = u_a.indices.size();
    cuembed::utils::RunTransposeReference<IndexT, int, ElemT>(
        options_,
        u_a.indices,
        u_a.offsets,
        u_a.weights,
        nnz_,
        &u_a.transpose_indices,
        &u_a.transpose_remapped_indices,
        &u_a.transpose_sample_ids,
        &u_a.transpose_weights);
  }

  void RunAddresserTest() {
    if (options_.combine_mode() == CombineMode::kSum) {
      RunReductionAddresserTest<CombineMode::kSum>();
    } else if (options_.combine_mode() == CombineMode::kMean) {
      RunReductionAddresserTest<CombineMode::kMean>();
    } else if (options_.combine_mode() == CombineMode::kConcat) {
      RunConcatAddresserTest();
    } else {
      CHECK(false);
    }
  }

  void RunCombinerTest() {
    if (options_.combine_mode() == CombineMode::kSum) {
      RunSumCombinerTest();
    } else if (options_.combine_mode() == CombineMode::kConcat) {
      RunConcatCombinerTest();
    } else if (options_.combine_mode() == CombineMode::kMean) {
      RunMeanCombinerTest();
    } else {
      CHECK(false);
    }
  }

  void RunIdxLoaderTest() {
    if (options_.is_csr()) {
      RunCSRIndexLoaderTest();
    } else {
      RunFixedHotnessTest();
    }
  }

  void RunGradAddresserTest() { RunGradAddresserTest_(); }
  void RunGradCombinerTest() { RunGradCombinerTest_(); }
  void RunGradIdxLoaderTest() { RunGradIdxLoaderTest_(); }

 private:
  void RunCSRIndexLoaderTest() {
    const int kNumHots = 0;
    for (int sample_id = 0; sample_id < options_.batch_size(); sample_id++) {
      IndexLoader<IndexT, ElemT, OffsetT> index_loader(options_.batch_size(),
                                                       sample_id,
                                                       u_a.indices.data().get(),
                                                       u_a.weights.data().get(),
                                                       u_a.offsets.data().get(),
                                                       kNumHots);
      auto index_start = u_a.offsets[sample_id];
      auto index_stop = u_a.offsets[sample_id + 1];
      CHECK_EQ(index_loader.GetHotness(), index_stop - index_start);

      for (int i_hotness = 0; i_hotness < index_loader.GetHotness();
           i_hotness++) {
        CHECK_EQ(index_loader.GetLookUpIndex(i_hotness),
                 u_a.indices[index_start + i_hotness]);
      }
    }
  }

  void RunFixedHotnessTest() {
    for (int sample_id = 0; sample_id < options_.batch_size(); sample_id++) {
      IndexLoader<IndexT, ElemT, void> index_loader(options_.batch_size(),
                                                    sample_id,
                                                    u_a.indices.data().get(),
                                                    u_a.weights.data().get(),
                                                    nullptr,
                                                    options_.hotness());
      CHECK_EQ(index_loader.GetHotness(), options_.hotness());
    }
  }
  template <CombineMode mode>
  void RunReductionAddresserTest() {
    int threads_per_sample = options_.embed_width() / elements_per_load_;
    for (int sample_id = 0; sample_id < options_.batch_size(); sample_id++) {
      for (int thread_idx = 0; thread_idx < threads_per_sample; thread_idx++) {
        int64_t embed_row_offset = thread_idx;
        int64_t output_row_offset = thread_idx;
        Addresser<ElemT, ElemT, IndexT, LoadVecT, LoadVecT, mode>
            test_addresser(u_a.embedding.data().get(),
                           u_a.result.data().get(),
                           sample_id,
                           options_.hotness(),
                           options_.embed_width());
        int hotness_for_sample = options_.hotness();
        int index_start = sample_id * options_.hotness();
        if (options_.is_csr()) {
          hotness_for_sample =
              u_a.offsets[sample_id + 1] - u_a.offsets[sample_id];
          index_start = u_a.offsets[sample_id];
        }
        for (int i = 0; i < hotness_for_sample; i++) {
          auto lookup_idx = u_a.indices[index_start + i];
          auto embedding_address =
              test_addresser.GetEmbeddingAddress(lookup_idx) + embed_row_offset;
          CHECK_EQ(reinterpret_cast<const void*>(
                       u_a.embedding.data().get() +
                       lookup_idx * options_.embed_width() +
                       thread_idx * elements_per_load_),
                   reinterpret_cast<const void*>(embedding_address));
        }
        auto output_address =
            test_addresser.GetOutputAddress() + output_row_offset;
        CHECK_EQ(reinterpret_cast<void*>(u_a.result.data().get() +
                                         sample_id * options_.embed_width() +
                                         thread_idx * elements_per_load_),
                 reinterpret_cast<void*>(output_address));
      }
    }
  }

  void RunConcatAddresserTest() {
    int threads_per_sample = options_.embed_width() / elements_per_load_;
    for (int sample_id = 0; sample_id < options_.batch_size(); sample_id++) {
      for (int thread_idx = 0; thread_idx < threads_per_sample; thread_idx++) {
        int64_t embed_row_offset = thread_idx;
        int64_t output_row_offset = thread_idx;
        Addresser<ElemT,
                  ElemT,
                  IndexT,
                  LoadVecT,
                  LoadVecT,
                  CombineMode::kConcat>
            test_addresser(u_a.embedding.data().get(),
                           u_a.result.data().get(),
                           sample_id,
                           options_.hotness(),
                           options_.embed_width());
        for (int i = 0; i < options_.hotness(); i++) {
          auto lookup_idx = u_a.indices[sample_id * options_.hotness() + i];
          auto embedding_address =
              test_addresser.GetEmbeddingAddress(lookup_idx) + embed_row_offset;
          auto output_address =
              test_addresser.GetConcatOutputAddress(i) + output_row_offset;
          CHECK_EQ(reinterpret_cast<const void*>(
                       u_a.embedding.data().get() +
                       lookup_idx * options_.embed_width() +
                       thread_idx * elements_per_load_),
                   reinterpret_cast<const void*>(embedding_address));
          CHECK_EQ(
              reinterpret_cast<void*>(
                  u_a.result.data().get() +
                  sample_id * options_.embed_width() * options_.hotness() +
                  options_.embed_width() * i + thread_idx * elements_per_load_),
              reinterpret_cast<void*>(output_address));
        }
        auto output_address =
            test_addresser.GetOutputAddress() + output_row_offset;
        CHECK_EQ(reinterpret_cast<void*>(u_a.result.data().get() +
                                         sample_id * options_.embed_width() *
                                             options_.hotness() +
                                         thread_idx * elements_per_load_),
                 reinterpret_cast<void*>(output_address));
      }
    }
  }

  template <CombineMode Mode, typename F>
  void ReductionCombinerTest(F ScaleIfNeeded) {
    Combiner<LoadVecT, ReduceVecT, LoadVecT, Mode> combiner;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> offset_distrib(
        0, options_.embed_width() / elements_per_load_);
    std::uniform_real_distribution<float> weight_distrib(0.0f, 1.0f);

    LoadVecT reference = GetZeroVec<LoadVecT>();
    LoadVecT output = GetZeroVec<LoadVecT>();
    float accumulated_weight = 0.f;
    for (int i = 0; i < options_.hotness(); i++) {
      auto lookup_offset = offset_distrib(gen);
      auto loading_address =
          reinterpret_cast<LoadVecT*>(u_a.embedding.data().get()) +
          lookup_offset;
      float weight = weight_distrib(gen);
      if (weight < 0.5f) {
        combiner.Gather(loading_address);
        reference += *(loading_address);
        accumulated_weight += 1.0f;
      } else {
        combiner.Gather(loading_address, weight);
        reference += (*loading_address) * weight;
        accumulated_weight += weight;
      }

      // Sum combiner does not output for concat.
      combiner.OutputForConcatIfNeeded(&output);
      EXPECT_TRUE(Equal(output, GetZeroVec<LoadVecT>()));
    }
    combiner.OutputForReductionIfNeeded(&output);
    EXPECT_TRUE(Equal(output, ScaleIfNeeded(reference, accumulated_weight)));
  }

  void RunSumCombinerTest() {
    ReductionCombinerTest<CombineMode::kSum>(
        [](const LoadVecT& reference, int hotness) { return reference; });
  }

  void RunMeanCombinerTest() {
    ReductionCombinerTest<CombineMode::kMean>(
        [](const LoadVecT& reference, float accumulated_weight) {
          return reference * (1.0f / accumulated_weight);
        });
  }

  void RunConcatCombinerTest() {
    Combiner<LoadVecT, ReduceVecT, LoadVecT, CombineMode::kConcat> combiner;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(
        0, options_.embed_width() / elements_per_load_);

    LoadVecT reference = GetZeroVec<LoadVecT>();
    LoadVecT output = GetZeroVec<LoadVecT>();
    for (int i = 0; i < options_.hotness(); i++) {
      auto lookup_offset = distrib(gen);
      auto loading_address =
          reinterpret_cast<LoadVecT*>(u_a.embedding.data().get()) +
          lookup_offset;
      combiner.Gather(loading_address);
      reference = *(loading_address);

      combiner.OutputForConcatIfNeeded(&output);
      EXPECT_TRUE(Equal(output, reference));
    }

    // Concat combiner does not write for reduction.
    output = GetZeroVec<LoadVecT>();
    reference = GetZeroVec<LoadVecT>();
    combiner.OutputForReductionIfNeeded(&output);
    EXPECT_TRUE(Equal(output, reference));
  }

  void RunGradAddresserTest_() {
    int threads_per_nz = options_.embed_width() / elements_per_load_;
    for (int nz_id = 0; nz_id < nnz_; nz_id += nz_block_size_) {
      for (int thread_idx = 0; thread_idx < threads_per_nz; thread_idx++) {
        int64_t embed_row_offset = thread_idx;
        GradAddresser<ElemT, LoadVecT> test_addresser(
            u_a.grad_y.data().get(),
            u_a.grad_embedding.data().get(),
            options_.embed_width());
        for (int i = 0; i < nz_block_size_; i++) {
          if (nz_id + i >= nnz_) continue;
          int sample_id = u_a.transpose_sample_ids[nz_id + i];
          int index = u_a.transpose_indices[nz_id + i];
          auto grad_y_address =
              test_addresser.GetGradResultAddress(sample_id) + embed_row_offset;
          auto grad_embedding_address =
              test_addresser.GetGradEmbeddingAddress(index) + embed_row_offset;
          CHECK_EQ(
              reinterpret_cast<const void*>(u_a.grad_y.data().get() +
                                            sample_id * options_.embed_width() +
                                            thread_idx * elements_per_load_),
              reinterpret_cast<const void*>(grad_y_address));
          CHECK_EQ(reinterpret_cast<void*>(u_a.grad_embedding.data().get() +
                                           index * options_.embed_width() +
                                           thread_idx * elements_per_load_),
                   reinterpret_cast<void*>(grad_embedding_address));
        }
      }
    }
  }

  void RunGradCombinerTest_() {
    GradCombiner<ElemT, LoadVecT, ElemT> grad_combiner;

    LoadVecT reference = GetZeroVec<LoadVecT>();
    LoadVecT output = GetZeroVec<LoadVecT>();
    unsigned int seed = 0;
    for (int i = 0; i < options_.hotness(); i++) {
      auto lookup_offset =
          rand() % (options_.embed_width() / elements_per_load_);
      auto loading_address =
          reinterpret_cast<LoadVecT*>(u_a.grad_y.data().get()) + lookup_offset;
      float weight =
          static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
      grad_combiner.Gather(loading_address, weight);
      reference += (*loading_address) * weight;

      // Sum combiner does not output for concat.
      bool should_write = ((rand_r(&seed) % 10) == 0);
      bool should_atomic = !should_write && ((rand_r(&seed) % 10) == 0);
      grad_combiner.WriteOrAtomic(&output, should_write, should_atomic);
      if (should_write || should_atomic) {
        EXPECT_TRUE(Equal(output, reference));
        reference = GetZeroVec<LoadVecT>();
        output = GetZeroVec<LoadVecT>();
      }
    }
  }

  void RunGradIdxLoaderTest_() {
    GradIndexLoader<IndexT, ElemT> grad_index_loader(
        u_a.transpose_indices.data().get(),
        u_a.transpose_sample_ids.data().get(),
        u_a.transpose_weights.data().get(),
        nnz_,
        nz_block_size_);
  }

  const int elements_per_load_ = sizeof(LoadVecT) / sizeof(ElemT);
  int nnz_;
  int nz_block_size_ = 64;

  utils::
      UniversalEmbeddingAllocation<ElemT, IndexT, OffsetT, ElemT, ElemT, ElemT>
          u_a;
  utils::AllocationOptions options_;
};

// TODO(zejiaz): separate out InputT apart from OutputT.
class OperationsTestEmbed32Vec4Idx32
    : public OperationsTest<float, float4, float4, int32_t, int> {};
class OperationsTestEmbed32Vec4Idx64
    : public OperationsTest<float, float4, float4, int64_t, int> {};
class OperationsTestEmbed16Vec4Idx32
    : public OperationsTest<__half, half4, half4, int32_t, int> {};
class OperationsTestEmbed16Vec4Idx64
    : public OperationsTest<__half, half4, half4, int64_t, int> {};
class OperationsTestEmbed32Vec2Idx32
    : public OperationsTest<float, float2, float2, int32_t, int> {};
class OperationsTestEmbed32Vec2Idx64
    : public OperationsTest<float, float2, float2, int64_t, int> {};
class OperationsTestEmbed16Vec2Idx32
    : public OperationsTest<__half, half2, half2, int32_t, int> {};
class OperationsTestEmbed16Vec2Idx64
    : public OperationsTest<__half, half2, half2, int64_t, int> {};

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
  std::string result = absl::StrFormat("Width%dBatch%dHot%d%s%s",
                                       options.embed_width(),
                                       options.batch_size(),
                                       options.hotness(),
                                       ToString(options.combine_mode()),
                                       options.is_csr() ? "CSR" : "FixedHot");
  return result;
}

TEST_P(OperationsTestEmbed32Vec4Idx32, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx32, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx32, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx64, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx64, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx64, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx32, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx32, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx32, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx64, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx64, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed16Vec4Idx64, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx32, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx32, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx32, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx64, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx64, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed32Vec2Idx64, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx32, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx32, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx32, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx64, TestAddresser) { RunAddresserTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx64, TestCombiner) { RunCombinerTest(); }
TEST_P(OperationsTestEmbed16Vec2Idx64, TestIdxLoader) { RunIdxLoaderTest(); }
TEST_P(OperationsTestEmbed32Vec4Idx32, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed32Vec4Idx32, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed32Vec4Idx32, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed32Vec4Idx64, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed32Vec4Idx64, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed32Vec4Idx64, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx32, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx32, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx32, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx64, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx64, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed16Vec4Idx64, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx32, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx32, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx32, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx64, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx64, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed32Vec2Idx64, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx32, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx32, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx32, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx64, TestGradAddresser) {
  RunGradAddresserTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx64, TestGradCombiner) {
  RunGradCombinerTest();
}
TEST_P(OperationsTestEmbed16Vec2Idx64, TestGradIdxLoader) {
  RunGradIdxLoaderTest();
}

// Some macros to make test allocation statement more concise.
#define ALLOC (utils::AllocationOptions().num_categories(10_K))
#define SUM CombineMode::kSum
#define CONCAT CombineMode::kConcat
#define AVG CombineMode::kMean
#define CSR is_csr(true)
auto lookup_test_values = testing::Values(
    ALLOC.batch_size(3).embed_width(4).hotness(4).combine_mode(SUM),
    ALLOC.batch_size(3).embed_width(4).hotness(4).combine_mode(SUM).CSR,
    ALLOC.batch_size(3).embed_width(4).hotness(4).combine_mode(AVG),
    ALLOC.batch_size(3).embed_width(4).hotness(4).combine_mode(AVG).CSR,
    ALLOC.batch_size(3).embed_width(4).hotness(4).combine_mode(CONCAT),
    ALLOC.batch_size(257).embed_width(36).hotness(26).combine_mode(SUM),
    ALLOC.batch_size(257).embed_width(36).hotness(26).combine_mode(SUM).CSR,
    ALLOC.batch_size(257).embed_width(36).hotness(26).combine_mode(AVG),
    ALLOC.batch_size(257).embed_width(36).hotness(26).combine_mode(AVG).CSR,
    ALLOC.batch_size(257).embed_width(36).hotness(26).combine_mode(CONCAT),
    ALLOC.batch_size(257).embed_width(68).hotness(33).combine_mode(SUM),
    ALLOC.batch_size(257).embed_width(68).hotness(33).combine_mode(SUM).CSR,
    ALLOC.batch_size(257).embed_width(68).hotness(33).combine_mode(AVG),
    ALLOC.batch_size(257).embed_width(68).hotness(33).combine_mode(AVG).CSR,
    ALLOC.batch_size(257).embed_width(68).hotness(33).combine_mode(CONCAT));

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed32Vec4Idx32,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed32Vec4Idx32::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed32Vec4Idx64,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed32Vec4Idx64::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed16Vec4Idx32,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed16Vec4Idx32::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed16Vec4Idx64,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed16Vec4Idx64::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed32Vec2Idx32,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed32Vec2Idx32::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed32Vec2Idx64,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed32Vec2Idx64::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed16Vec2Idx32,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed16Vec2Idx32::ParamType>&
           info) { return GenerateTestName(info.param); });

INSTANTIATE_TEST_SUITE_P(
    OperationsTest,
    OperationsTestEmbed16Vec2Idx64,
    lookup_test_values,
    [](const testing::TestParamInfo<OperationsTestEmbed16Vec2Idx64::ParamType>&
           info) { return GenerateTestName(info.param); });
}  // namespace cuembed
