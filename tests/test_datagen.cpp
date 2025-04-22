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

#include <gtest/gtest.h>

#include "utils/include/datagen.h"

namespace cuembed {
namespace index_generators {

template <typename T>
class DataGeneratorTest : public ::testing::Test {
 protected:
  using IndexType = T;

  void setUpParams(const int num_categories, const int num_hot) {
    num_categories_ = num_categories;
    num_hot_ = num_hot;
  }

  std::vector<std::vector<T>> generateIndices(const int batch_size) {
    std::vector<std::vector<T>> result;
    for (int i = 0; i < batch_size; i++) {
      std::vector<T> generated_idx = generator_->getCategoryIndices();
      EXPECT_EQ(generated_idx.size(), num_hot_);
      result.push_back(generated_idx);
    }
    return result;
  }

  void sanityCheckIndices(const std::vector<std::vector<T>>& indices) {
    for (const auto batch : indices) {
      std::set<T> used_indices;
      for (const auto index : batch) {
        EXPECT_GT(index, 0);
        EXPECT_LE(index, num_categories_);
        // Checks that the indices are generated with no repetitions.
        EXPECT_EQ(used_indices.count(index), 0);
        used_indices.insert(index);
      }
    }
  }

  std::vector<int> computeHistogram(
      const std::vector<std::vector<T>>& generated_indices) {
    std::vector<int> result(num_categories_ + 1, 0);
    for (const auto& sample : generated_indices) {
      for (const auto& index : sample) {
        result[index]++;
      }
    }
    return result;
  }

  void resetGenerator(FeatureGenerator<T>* new_generator) {
    generator_.reset(new_generator);
  }

  // Generates indices according to the given batchsize.
  // Computes histogram of the generated indices.
  // Checks that the computed histogram normalized to probability distribution
  // is within delta compared to the provided expected probability.
  void checkGeneratorMatchesExpectation(
      const int batch_size,
      const std::vector<double>& expected_probability,
      const double delta = 1e-4) {
    const auto histogram =
        this->computeHistogram(this->generateIndices(batch_size));
    EXPECT_EQ(histogram.size(), expected_probability.size());

    for (size_t i = 0; i < histogram.size(); i++) {
      if (i == 0) {
        // Category feature 0 is reserved.
        EXPECT_EQ(histogram[i], 0);
      } else {
        EXPECT_NEAR(
            static_cast<double>(histogram[i]) / static_cast<double>(batch_size),
            expected_probability[i],
            delta);
      }
    }
  }

 private:
  std::unique_ptr<FeatureGenerator<T>> generator_;
  int num_categories_;
  int num_hot_;
};

TYPED_TEST_SUITE_P(DataGeneratorTest);

// Test that psx power law generator follows the analytical distribution.
TYPED_TEST_P(DataGeneratorTest, OneHotPsxPowerLawGenerationWorks) {
  using IndexType = typename TestFixture::IndexType;
  const int kNumCategories = 9;
  const int kNumHot = 1;
  const int kBatchSize = 4000000;
  const double kAlpha = 1.15;
  std::vector<double> expected_probability(kNumCategories + 1, 0);

  double sum = 0;
  for (size_t i = 1; i < expected_probability.size(); i++) {
    // f(i) = i ^(-alpha).
    // expected_probability(i) = \integral_(i)^(i+1){f(x)dx}.
    expected_probability[i] =
        (-kAlpha) * pow(static_cast<double>(i), (1 - kAlpha)) -
        (-kAlpha) * pow(static_cast<double>(i + 1), (1 - kAlpha));
    sum += expected_probability[i];
  }
  // Normalize the probability distribution.
  for (auto& prob : expected_probability) {
    prob /= sum;
  }

  this->setUpParams(kNumCategories, kNumHot);
  this->resetGenerator(
      new index_generators::PowerLawFeatureGenerator<IndexType>(
          kNumCategories, kNumHot, kAlpha));

  const double delta = 1e-3;  // allow 0.1% off from expected.
  this->checkGeneratorMatchesExpectation(
      kBatchSize, expected_probability, delta);
}

// Check that multi-hot generators do not have repetitions.
// Check that all indices generated are within range [1, num_category - 1].
TYPED_TEST_P(DataGeneratorTest, MultiHotGenerationWorks) {
  using IndexType = typename TestFixture::IndexType;
  const int kNumCategories = 1000;
  const int kNumHot = 64;
  const int kBatchSize = 40000;
  const double kAlpha = 1.15;

  std::vector<index_generators::FeatureGenerator<IndexType>*> generators({
      new index_generators::PowerLawFeatureGenerator<IndexType>(
          kNumCategories, kNumHot, kAlpha, false, false, PowerLawType::kPsx),
  });

  for (auto& generator : generators) {
    this->setUpParams(kNumCategories, kNumHot);
    this->resetGenerator(generator);
    this->sanityCheckIndices(this->generateIndices(kBatchSize));
  }
}

REGISTER_TYPED_TEST_SUITE_P(DataGeneratorTest,
                            OneHotPsxPowerLawGenerationWorks,
                            MultiHotGenerationWorks);

using Types = ::testing::Types<int32_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(DataGen, DataGeneratorTest, Types);

}  // namespace index_generators
}  // namespace cuembed
