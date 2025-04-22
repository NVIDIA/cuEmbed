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

file(GLOB test_source_files_cpp "${CMAKE_SOURCE_DIR}/tests/*.cpp")
file(GLOB test_source_files_cu  "${CMAKE_SOURCE_DIR}/tests/*.cu")
set(test_source_files ${test_source_files_cpp} ${test_source_files_cu})

foreach(test_file ${test_source_files})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_include_directories(${test_name} PRIVATE googletest ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(
        ${test_name} PRIVATE
        cuembed_hdrs
        gtest
        gtest_main
        cuda
        utils
        absl::log 
        absl::check)
    set_target_properties(${test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/test"
    )
endforeach(test_file)
