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

#ifndef UTILS_INCLUDE_DATAGEN_H_
#define UTILS_INCLUDE_DATAGEN_H_

#include <memory>
#include <random>
#include <set>
#include <vector>

namespace cuembed {
namespace index_generators {

// Abstract base class for generators of feature categories. Derived classes
// will be specialized to draw random values from particular distributions. Any
// generator requires two parameters for the features: number of categories and
// the number of indices per sample (aka hotness). Each call to the
// getCategoryIndices() method return a C++ vector with the randomly generated
// category indices. There will be no index repetitions in this returned vector.
template <typename IndexType = int>
class FeatureGenerator {
 public:
  // Deleted to ensure that parameters are always initialized upon construction.
  FeatureGenerator() = delete;

  // Constructs an object given the number of categories and hotness for a
  // feature.
  FeatureGenerator(const IndexType num_categories,
                   const int num_hot,
                   const bool shuffle = false,
                   const bool permute = false);

  virtual ~FeatureGenerator() {}

  // Each derived feature generator should implement this method.
  virtual IndexType generateIndex() = 0;

  // Returns a vector of random category indices.
  std::vector<IndexType> getCategoryIndices();

  // Returns the number of categories for the feature.
  int getNumCategories() const { return num_categories_; }

  // Returns the hotness for teh feature (number of looks ups per sample).
  size_t getNumHot() const { return static_cast<size_t>(num_hot_); }

  IndexType getPermutedIndex(int index) const;

  const std::vector<IndexType>& getInversePermutation() const {
    return this->inverse_permutation_;
  }

 protected:
  IndexType num_categories_ = 0;
  int num_hot_ = 0;
  bool shuffle_ = false;
  bool permute_ = false;
  std::vector<IndexType> permutation_;
  std::vector<IndexType> inverse_permutation_;
  std::default_random_engine rng_;
};

// PowerLaw type specifies the type of power law distribution used in the
// generator.
enum class PowerLawType {
  // Generate random indices in [1, num_categories] range according to power
  // law.
  kPsx = 0
};

// A class for generating category indices drawn from a power law distribution.
// Category index 0 is not generated as it is assumed to be reserved for a
// "missing" category. Thus, given num_categories, returned indices are drawn
// from [1, num_categories] range. Each returned set of indices contains exactly
// num_hots indices, with no repetitions. Power law distribution is specified
// via its exponent, alpha > 0. Smaller indices correspond to more frequent
// categories (i.e. 1 will be the most frequent category, 2 - the second most
// frequent one, etc.). If math_numpy is true, generate distribution that
// matches numpy.
template <typename IndexType = int>
class PowerLawFeatureGenerator : public FeatureGenerator<IndexType> {
 public:
  PowerLawFeatureGenerator() = delete;

  PowerLawFeatureGenerator(const IndexType num_categories,
                           const int num_hot,
                           const double alpha,
                           const bool shuffle = false,
                           const bool permute = false,
                           const PowerLawType type = PowerLawType::kPsx);

  IndexType generateIndex() override;

 protected:
  double alpha_ = 0.;  // Exponent for the power-law distribution.
  PowerLawType type_ = PowerLawType::kPsx;
  std::default_random_engine generator_;
  std::unique_ptr<std::uniform_real_distribution<double>> distribution_;
};
}  // namespace index_generators
}  // namespace cuembed

#endif  // UTILS_INCLUDE_DATAGEN_H_
