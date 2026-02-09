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

#include "utils/include/datagen.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

namespace cuembed {
namespace index_generators {

// Function that "translates" a value drawn uniformly from [0, 1) range into a
// value drawn from a power-law distribution. Power-law distribution is
// characterized by the range of values [min_value, max_value) and the exponent
// value alpha. Assumptions:
//  * k_min >= 1
//  * alpha > 0 && alpha != 1
// See the accompanying derivation.jpg file for a derivation of the equation
// used in this function. TODO: check if max_value can really be generated or if
// returned values are in [min_value, max_value) range.
template <typename Type = double>
float translateToPowerLaw(const Type min_value,
                          const Type max_value,
                          const Type alpha,
                          const Type random_uniform_value) {
  const Type gamma = 1 - alpha;
  Type y = pow(
      random_uniform_value * (pow(max_value, gamma) - pow(min_value, gamma)) +
          pow(min_value, gamma),
      1.0 / gamma);
  return y;
}

template <typename IndexType>
FeatureGenerator<IndexType>::FeatureGenerator(const IndexType num_categories,
                                              const int num_hot,
                                              const bool shuffle,
                                              const bool permute)
    : num_categories_(num_categories),
      num_hot_(num_hot),
      shuffle_(shuffle),
      permute_(permute) {
  // 0 is reserved. Need at least one additional category to generate indices
  // from.
  assert(this->num_categories_ > 1);
  if (permute) {
    this->permutation_.resize(num_categories + 1);
    this->inverse_permutation_.resize(num_categories + 1);
    std::iota(this->permutation_.begin(), this->permutation_.end(), 0);
    std::shuffle(
        this->permutation_.begin(), this->permutation_.end(), this->rng_);

    for (IndexType i = 0; i < num_categories + 1; ++i) {
      this->inverse_permutation_[this->permutation_[i]] = i;
    }
  }
}

template <typename IndexType>
IndexType FeatureGenerator<IndexType>::getPermutedIndex(int index) const {
  if (this->permute_) {
    return this->permutation_[index];
  } else {
    return index;
  }
}

template <typename IndexType>
std::vector<IndexType> FeatureGenerator<IndexType>::getCategoryIndices() {
  // A set created to track already used indices.
  std::set<IndexType> used_indices;
  while (used_indices.size() < this->getNumHot()) {
    used_indices.insert(this->getPermutedIndex(this->generateIndex()));
  }

  std::vector<IndexType> indices;
  for (const auto& x : used_indices) {
    indices.push_back(x);
  }

  if (this->shuffle_) {
    std::shuffle(indices.begin(), indices.end(), this->rng_);
  }

  return indices;
}

template <typename IndexType>
PowerLawFeatureGenerator<IndexType>::PowerLawFeatureGenerator(
    const IndexType num_categories,
    const int num_hot,
    const double alpha,
    const bool shuffle,
    const bool permute,
    const PowerLawType type)
    : FeatureGenerator<IndexType>(num_categories, num_hot, shuffle, permute),
      alpha_(alpha),
      type_(type) {
  distribution_.reset(new std::uniform_real_distribution<double>(0., 1.));
}

template <typename IndexType>
IndexType PowerLawFeatureGenerator<IndexType>::generateIndex() {
  const double x = (*distribution_)(generator_);
  IndexType y = -1;

  // translateToPowerLaw(1., num_categories + 1, *, *) generates an index
  // within range [1, num_categories + 1). Then cast to IndexType to range [1,
  // num_categories].
  y = IndexType(translateToPowerLaw(
      1., static_cast<double>(this->num_categories_ + 1), alpha_, x));

  return y;
}  // namespace index_generators

template class FeatureGenerator<int32_t>;
template class FeatureGenerator<int64_t>;
template class PowerLawFeatureGenerator<int32_t>;
template class PowerLawFeatureGenerator<int64_t>;

}  // namespace index_generators
}  // namespace cuembed
