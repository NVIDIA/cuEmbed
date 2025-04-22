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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/unique.h>
#include <thrust/universal_vector.h>

#include <fstream>
#include <type_traits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cuembed/include/embedding_lookup.cuh"
#include "cuembed/include/index_transforms.cuh"
#include "utils/include/datagen.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// clang-format off
ABSL_FLAG(int, num_categories, 1048576,
          "Number of categories/rows of embedding");
ABSL_FLAG(int, embed_width, 128,
          "Width of embedding vector");
ABSL_FLAG(int, batch_size, 1024,
          "Batch size");
ABSL_FLAG(int, hotness, 1,
          "Number of nonzero indices per sample");
ABSL_FLAG(int, iterations, 1,
          "Number of iterations to run benchmark");
ABSL_FLAG(float, alpha, 0.,
          "alpha of power distribution. Use uniform if alpha is 0");
ABSL_FLAG(bool, use_int64_indices, false,
          "If true, use int64_t type for lookup indices.");
ABSL_FLAG(bool, check_result, false,
          "If true, compare GPU result against CPU reference");
ABSL_FLAG(bool, half_embedding_type, false,
          "If true, use fp16 for embedding");
ABSL_FLAG(bool, csr_input, false,
          "If true, use CSR formats for embedding lookup indices");
ABSL_FLAG(bool, weighted_sum, false,
          "If true, summation of rows would be weighted");
ABSL_FLAG(bool, fp16_math, false,
          "If true, fp16 embed rows will be use fp16 math during reduction."
          "This flag has no effect when the embed rows are in fp32.");
ABSL_FLAG(bool, compressed_grad, true,
          "If true, will compute a sparse gradient in the backward pass.");
ABSL_FLAG(bool, skip_grad_init, true,
          "If true, will skip the zero-initializion of the gradient "
          "during backward.");
ABSL_FLAG(bool, forward_only, false,
          "If true, will run only forward, skipping transpose and backward.");
ABSL_FLAG(bool, enable_csv, false,
          "If true, will output results in CSV format.");
ABSL_FLAG(bool, enable_stderr, true,
          "If true, will set stderr log level to INFO.");
ABSL_FLAG(bool, clear_caches, true,
          "If true, will clear caches between invocations by summing the "
          "full embedding table.");
// clang-format on

template <typename T>
void ValidateResult(const thrust::universal_vector<T>& result,
                    const thrust::universal_vector<T>& h_result) {
  CHECK_EQ(thrust::equal(result.begin(), result.end(), h_result.begin()), true);
  CHECK_EQ(result.size(), h_result.size());
}

std::string combine_mode_str(cuembed::CombineMode mode) {
  switch (mode) {
    case cuembed::CombineMode::kSum:
      return "kSum";
    case cuembed::CombineMode::kMean:
      return "kMean";
    case cuembed::CombineMode::kConcat:
      return "kConcat";
  }
  return "Unknown";
}

void dump_csv_header(std::ofstream& outfile) {
  outfile << "num_categories,batch_size,hotness,alpha,embed_width,combine_mode,"
             "is_csr,is_weighted,compressed_grad,skip_grad_init,name,"
             "iterations,elapsed_time_ms,avg_time_ms,algo_bw_l2,algo_bw_dram"
          << std::endl;
}

void dump_csv_line(std::ofstream& outfile,
                   const cuembed::utils::AllocationOptions& options,
                   std::string name,
                   int iterations,
                   double elapsed_time_ms,
                   double algo_bw_l2,
                   double algo_bw_dram) {
  outfile << options.num_categories() << "," << options.batch_size() << ","
          << options.hotness() << "," << options.alpha() << ","
          << options.embed_width() << ","
          << combine_mode_str(options.combine_mode()) << "," << options.is_csr()
          << "," << options.is_weighted() << "," << options.compressed_grad()
          << "," << options.skip_grad_init() << "," << name << ","
          << absl::StrFormat("%d ", iterations) << ","
          << absl::StrFormat("%.2f ", elapsed_time_ms) << ","
          << absl::StrFormat("%.2f ", elapsed_time_ms / iterations) << ","
          << absl::StrFormat("%.2f", algo_bw_l2) << ","
          << absl::StrFormat("%.2f", algo_bw_dram) << std::endl;
}

bool file_exists(const std::string& fname) {
  std::ifstream infile(fname);
  return infile.good();
}

template <typename ElemT>
void clear_cache(ElemT* clear_cache_max,
                 const thrust::device_vector<int>& clear_cache_buffer) {
  *clear_cache_max += thrust::reduce(thrust::device,
                                     clear_cache_buffer.begin(),
                                     clear_cache_buffer.end(),
                                     0,
                                     thrust::maximum<ElemT>());
}

namespace cuembed {
template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void EmbeddingLookupBenchmark(const int num_categories,
                              const int embed_width,
                              const int batch_size,
                              const int hotness,
                              const float alpha,
                              const bool is_csr,
                              const bool is_weighted,
                              const bool compressed_grad,
                              const bool skip_grad_init,
                              const bool forward_only,
                              const bool check_result,
                              const int iterations,
                              const bool enable_csv,
                              const bool clear_caches) {
  utils::AllocationOptions options;
  options.num_categories(num_categories)
      .batch_size(batch_size)
      .hotness(hotness)
      .alpha(alpha)
      .embed_width(embed_width)
      .combine_mode(CombineMode::kSum)
      .is_csr(is_csr)
      .is_weighted(is_weighted)
      .compressed_grad(compressed_grad)
      .skip_grad_init(skip_grad_init);

  std::ofstream outfile;
  if (enable_csv) {
    std::string fname = "manual_benchmark_out.csv";
    bool existed_before = file_exists(fname);
    outfile.open(fname, std::ios::out | std::ios::app);

    if (!outfile.is_open()) {
      std::cerr << "Unable to open file\n";
      return;
    }

    if (!existed_before) {
      dump_csv_header(outfile);
    }
  }

  // Allocate buffers
  utils::
      UniversalEmbeddingAllocation<ElemT, IndexT, OffsetT, ElemT, ElemT, ElemT>
          u_a;
  utils::DeviceEmbeddingAllocation<ElemT, IndexT, OffsetT, ElemT, ElemT, ElemT>
      d_a;
  utils::AllocateHost(options, &u_a, forward_only);
  utils::AllocateDevice(options, u_a, &d_a, forward_only);

  // Used for clearing caches
  thrust::device_vector<int> clear_cache_buffer;
  if (clear_caches) {
    clear_cache_buffer.resize(256000000L, 1);
  }
  ElemT clear_cache_max = static_cast<ElemT>(0);

  // Warm up
  utils::RunForward<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                       d_a.embedding,
                                                       d_a.indices,
                                                       d_a.offsets,
                                                       d_a.weights,
                                                       &d_a.result);

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  // Actual run and recording elapsed time.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time_ms = 0.0;

  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    utils::RunForward<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                         d_a.embedding,
                                                         d_a.indices,
                                                         d_a.offsets,
                                                         d_a.weights,
                                                         &d_a.result);

    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));

      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw = 0.0;
  if (options.is_csr()) {
    algo_bw = sizeof(ElemT) * iterations *
              (d_a.offsets.back() - 1 + options.batch_size()) *
              options.embed_width() / 1.e6 / elapsed_time_ms;
  } else {
    algo_bw = sizeof(ElemT) * iterations * options.batch_size() *
              (options.hotness() + (options.combine_mode() == CombineMode::kSum
                                        ? 1
                                        : options.hotness())) *
              options.embed_width() / 1.e6 / elapsed_time_ms;
  }

  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "forward",
                  iterations,
                  elapsed_time_ms,
                  algo_bw,
                  0.0 /*algo_bw_dram*/);
  }
  LOG(INFO) << "Embedding forward. Iterations: "
            << absl::StrFormat("%d ", iterations) << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms) << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms / iterations)
            << ", Application BW [GB/s]: " << absl::StrFormat("%.2f", algo_bw);

  if (check_result) {
    utils::RunForwardReference<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                                  u_a.embedding,
                                                                  u_a.indices,
                                                                  u_a.offsets,
                                                                  u_a.weights,
                                                                  &u_a.result);
    ValidateResult<ElemT>(d_a.result, u_a.result);
    LOG(INFO) << "Check result forward passed";
  }

  if (forward_only) {
    return;
  }

  OffsetT nnz = static_cast<OffsetT>(d_a.indices.size());
  utils::RunTranspose<IndexT, OffsetT, ElemT>(options,
                                              d_a.indices,
                                              d_a.offsets,
                                              d_a.weights,
                                              nnz,
                                              &d_a.transpose_indices,
                                              &d_a.transpose_remapped_indices,
                                              &d_a.transpose_sample_ids,
                                              &d_a.transpose_weights,
                                              &d_a.sample_ids,
                                              &d_a.transpose_workspace);

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  float elapsed_time_ms_transpose = 0.0;
  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    utils::RunTranspose<IndexT, OffsetT, ElemT>(options,
                                                d_a.indices,
                                                d_a.offsets,
                                                d_a.weights,
                                                nnz,
                                                &d_a.transpose_indices,
                                                &d_a.transpose_remapped_indices,
                                                &d_a.transpose_sample_ids,
                                                &d_a.transpose_weights,
                                                &d_a.sample_ids,
                                                &d_a.transpose_workspace);
    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));
      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms_transpose += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw_transpose = 0.0;

  // Input
  algo_bw_transpose += nnz * sizeof(IndexT);
  algo_bw_transpose += (options.is_csr()) ? nnz * sizeof(OffsetT) : 0;
  algo_bw_transpose += (options.is_weighted()) ? nnz * sizeof(ElemT) : 0;

  // Output
  algo_bw_transpose +=
      ((options.compressed_grad()) ? 3 : 2) * nnz * sizeof(IndexT);
  algo_bw_transpose += (options.is_weighted()) ? nnz * sizeof(ElemT) : 0;

  algo_bw_transpose *= iterations;
  algo_bw_transpose /= (1.e6);
  algo_bw_transpose /= elapsed_time_ms_transpose;
  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "transpose",
                  iterations,
                  elapsed_time_ms_transpose,
                  0.0,
                  algo_bw_transpose);
  }

  LOG(INFO) << "Transpose. Iterations: " << absl::StrFormat("%d ", iterations)
            << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_transpose)
            << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_transpose / iterations)
            << ", Application BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_transpose);

  if (check_result) {
    utils::RunTransposeReference<IndexT, OffsetT, ElemT>(
        options,
        u_a.indices,
        u_a.offsets,
        u_a.weights,
        nnz,
        &u_a.transpose_indices,
        &u_a.transpose_remapped_indices,
        &u_a.transpose_sample_ids,
        &u_a.transpose_weights);
    ValidateResult<IndexT>(d_a.transpose_indices, u_a.transpose_indices);
    if (options.compressed_grad()) {
      ValidateResult<IndexT>(d_a.transpose_remapped_indices,
                             u_a.transpose_remapped_indices);
    }
    LOG(INFO) << "Check results transpose passed";
  }

  int num_unique = (options.compressed_grad())
                       ? d_a.transpose_remapped_indices.back() + 1
                       : 0;

  utils::RunBackward<ElemT, IndexT, OffsetT>(options,
                                             d_a.grad_y,
                                             d_a.transpose_indices,
                                             d_a.transpose_remapped_indices,
                                             d_a.transpose_sample_ids,
                                             d_a.transpose_weights,
                                             d_a.offsets,
                                             nnz,
                                             num_unique,
                                             &d_a.grad_embedding,
                                             &d_a.inverse_mapping);

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  float elapsed_time_ms_backward = 0.0;
  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    utils::RunBackward<ElemT, IndexT, OffsetT>(options,
                                               d_a.grad_y,
                                               d_a.transpose_indices,
                                               d_a.transpose_remapped_indices,
                                               d_a.transpose_sample_ids,
                                               d_a.transpose_weights,
                                               d_a.offsets,
                                               nnz,
                                               num_unique,
                                               &d_a.grad_embedding,
                                               &d_a.inverse_mapping);

    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));

      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms_backward += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw_backward_dram = 0.0;

  // Need to compute number of unique indices for bandwidth calcs
  int num_unique_ = thrust::unique_count(thrust::device,
                                         d_a.transpose_indices.begin(),
                                         d_a.transpose_indices.begin() + nnz);

  // Writes to embedding weight gradient
  algo_bw_backward_dram += sizeof(ElemT) * options.embed_width() * num_unique_;

  // Reads of COO lookup indices
  algo_bw_backward_dram += sizeof(IndexT) * nnz * 2;
  algo_bw_backward_dram += (options.is_weighted()) ? sizeof(ElemT) * nnz : 0;

  // Reads from grad_y
  double algo_bw_backward_l2 = 0.0;
  if (options.combine_mode() == CombineMode::kConcat) {
    algo_bw_backward_dram += sizeof(ElemT) * options.embed_width() * nnz;
    algo_bw_backward_l2 = algo_bw_backward_dram;
  } else {
    algo_bw_backward_dram +=
        sizeof(ElemT) * options.embed_width() * options.batch_size();
    algo_bw_backward_l2 =
        algo_bw_backward_dram + sizeof(ElemT) * options.embed_width() * nnz;
  }

  algo_bw_backward_dram =
      (algo_bw_backward_dram * iterations) / 1e6 / elapsed_time_ms_backward;
  algo_bw_backward_l2 =
      (algo_bw_backward_l2 * iterations) / 1e6 / elapsed_time_ms_backward;

  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "backward",
                  iterations,
                  elapsed_time_ms_backward,
                  algo_bw_backward_l2,
                  algo_bw_backward_dram);
  }

  LOG(INFO) << "Backward. Iterations: " << absl::StrFormat("%d ", iterations)
            << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_backward)
            << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_backward / iterations)
            << ", Application DRAM BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_backward_dram)
            << ", Application L2 BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_backward_l2);

  if (check_result) {
    utils::RunBackwardReference<ElemT, IndexT, OffsetT>(
        options,
        u_a.grad_y,
        u_a.transpose_indices,
        u_a.transpose_remapped_indices,
        u_a.transpose_sample_ids,
        u_a.transpose_weights,
        u_a.offsets,
        nnz,
        &u_a.grad_embedding,
        &u_a.inverse_mapping);
    ValidateResult<ElemT>(d_a.grad_embedding, u_a.grad_embedding);
    if (options.compressed_grad()) {
      ValidateResult<IndexT>(d_a.inverse_mapping, u_a.inverse_mapping);
    }
    LOG(INFO) << "Check result backward passed";
  }
  if (enable_csv) {
    outfile.close();
  }
}

}  // namespace cuembed

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  int num_categories = absl::GetFlag(FLAGS_num_categories);
  int embed_width = absl::GetFlag(FLAGS_embed_width);
  int batch_size = absl::GetFlag(FLAGS_batch_size);
  int hotness = absl::GetFlag(FLAGS_hotness);
  int iterations = absl::GetFlag(FLAGS_iterations);
  float alpha = absl::GetFlag(FLAGS_alpha);
  bool check_result = absl::GetFlag(FLAGS_check_result);
  bool half_embedding_type = absl::GetFlag(FLAGS_half_embedding_type);
  bool use_int64_indices = absl::GetFlag(FLAGS_use_int64_indices);
  bool is_csr = absl::GetFlag(FLAGS_csr_input);
  bool is_weighted = absl::GetFlag(FLAGS_weighted_sum);
  bool fp16_math = absl::GetFlag(FLAGS_fp16_math);
  bool compressed_grad = absl::GetFlag(FLAGS_compressed_grad);
  bool skip_grad_init = absl::GetFlag(FLAGS_skip_grad_init);
  bool forward_only = absl::GetFlag(FLAGS_forward_only);
  bool enable_csv = absl::GetFlag(FLAGS_enable_csv);
  bool enable_stderr = absl::GetFlag(FLAGS_enable_stderr);
  bool clear_caches = absl::GetFlag(FLAGS_clear_caches);
  LOG(INFO) << "parsed flag num_categories: " << num_categories
            << ", embed_width: " << embed_width
            << ", batch_size: " << batch_size << ", hotness: " << hotness
            << ", alpha: " << alpha
            << ", fp16 embedding: " << half_embedding_type
            << ", int64_t indices: " << use_int64_indices
            << ", csr indices: " << is_csr << ", weighted sum: " << is_weighted
            << ", fp16 math: " << fp16_math
            << ", sparse gradient: " << compressed_grad
            << ", skip gradient init: " << skip_grad_init
            << ", forward_only: " << forward_only
            << ", enable_csv: " << enable_csv
            << ", enable_stderr: " << enable_stderr
            << ", clear_caches: " << clear_caches;

  if (enable_stderr) {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  } else {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);
  }

  if (half_embedding_type && use_int64_indices && fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int64_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  } else if (half_embedding_type && !use_int64_indices && fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int32_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  } else if (half_embedding_type && use_int64_indices && !fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int64_t, int, false>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  } else if (half_embedding_type && !use_int64_indices && !fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int32_t, int, false>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  } else if (!half_embedding_type && use_int64_indices) {
    cuembed::EmbeddingLookupBenchmark<float, int64_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  } else if (!half_embedding_type && !use_int64_indices) {
    cuembed::EmbeddingLookupBenchmark<float, int32_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches);
  }

  return 0;
}
