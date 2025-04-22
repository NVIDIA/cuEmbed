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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "utils/include/datagen.h"

TEST(GtestTest, BasicTest) { EXPECT_EQ(1, 1); }

TEST(LoggingTest, BasicLogging) {
  // Redirect glog output to prevent it from polluting test output
  EXPECT_NO_FATAL_FAILURE(LOG(INFO) << "This is an info log for testing.");

  // Test other logging levels.
  EXPECT_NO_FATAL_FAILURE(LOG(ERROR) << "This is an error log for testing.");
}

TEST(CheckTest, BasicCheck) {
  EXPECT_NO_FATAL_FAILURE(CHECK_EQ(1, 1));
  EXPECT_DEATH(CHECK_EQ(1, 2), "");
}

TEST(Thrust, BasicTest) {
  thrust::host_vector<int> h_vec{1, 3, 2, 4};
  thrust::device_vector<int> d_vec = h_vec;
  thrust::sort(d_vec.begin(), d_vec.end());
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  for (size_t i = 0; i < h_vec.size(); i++) {
    EXPECT_EQ(h_vec[i], i + 1);
  }
}
