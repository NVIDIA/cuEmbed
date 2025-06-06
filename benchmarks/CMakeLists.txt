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

find_package(CUDAToolkit REQUIRED)

add_executable(manual_benchmark manual_benchmark.cu)
target_include_directories(manual_benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(
  manual_benchmark PRIVATE 
  cuembed_hdrs
  cuda 
  utils 
  absl::log 
  absl::log_initialize
  absl::check
  absl::flags 
  absl::flags_parse)
set_target_properties(manual_benchmark
  PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/benchmarks")
