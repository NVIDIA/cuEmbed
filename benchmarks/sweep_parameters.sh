#!/bin/bash

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

rm -f manual_benchmark_out.csv 

benchmark=${1:-../build/bin/benchmarks/manual_benchmark}
for alpha in 0.0 1.05 1.15
do
  for num_categories in 1000000 10000000
  do
    for embed_width in 32 128
    do
      for batch in 1024 32768 131072
      do
        for hotness in 1 16 64 
        do
            ${benchmark} --num_categories "${num_categories}" --embed_width "${embed_width}" --batch_size "${batch}" --alpha=${alpha} --hotness="${hotness}" --iterations=1000 --enable_csv
        done
      done
    done
  done
done
