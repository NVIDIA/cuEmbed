# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
project(cuembed CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

set(ABSL_PROPAGATE_CXX_STD ON)

option(BUILD_TESTS "Build the tests" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

find_package(CUDAToolkit)

set(CUEMBED_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(cuembed_hdrs INTERFACE ${cuembed_source_files})
target_include_directories(cuembed_hdrs INTERFACE ${CUEMBED_PROJECT_SOURCE_DIR})
add_library(cuembed::hdrs ALIAS cuembed_hdrs)

if (BUILD_TESTS OR BUILD_BENCHMARKS)
    # TODO(zejiaz): move to CPM instead of submodule
    add_subdirectory(third_party/abseil-cpp)

    # Utility library for benchmarking and testing.
    add_subdirectory(utils)
endif()

# Setup tests
if(BUILD_TESTS)
    add_subdirectory(third_party/gtest)
    add_subdirectory(tests)
endif()

# Benchmarks.
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Examples
if (BUILD_EXAMPLES)
    add_subdirectory(examples/pytorch)
endif()