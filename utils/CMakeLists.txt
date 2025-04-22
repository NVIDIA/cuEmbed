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

include_directories(${CUEMBED_PROJECT_SOURCE_DIR})

find_package(CUDAToolkit)
add_library(utils OBJECT src/embedding_allocation.cu src/embedding_gpu_forward.cu src/embedding_gpu_transpose.cu src/embedding_gpu_backward.cu src/embedding_cpu.cu src/datagen.cpp)
target_include_directories(utils PRIVATE ${CUDAToolkit_INCLUDE_DIRS} absl::log absl::check gtest)
target_link_libraries(utils PRIVATE CUDA::cudart absl::log absl::check gtest)
