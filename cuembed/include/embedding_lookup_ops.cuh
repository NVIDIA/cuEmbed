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

//! \file
#ifndef CUEMBED_INCLUDE_EMBEDDING_LOOKUP_OPS_CUH_
#define CUEMBED_INCLUDE_EMBEDDING_LOOKUP_OPS_CUH_

// clang-format off
#include <cuda_pipeline_primitives.h>
#include <cuda/std/type_traits>
// clang-format on

#include <algorithm>

#include "cuembed/include/embedding_lookup_types.cuh"

#ifndef __CUDA_ARCH__
#define FOR_HOST_TEST
#endif

template <typename IndexT, typename WeightT>
__device__ __host__ __forceinline__ int GetSmemBytesPerNzBlock(
    const int nz_block_size) {
  const int max_elt_size = max(sizeof(IndexT), sizeof(WeightT));
  const int unpadded_bytes = nz_block_size * (sizeof(IndexT) +   // indices
                                              sizeof(IndexT) +   // sample_ids
                                              sizeof(WeightT) +  // weights
                                              sizeof(bool) +     // should_write
                                              sizeof(bool));  // should_atomic
  const int sm_width = 128;
  const int padded_bytes =
      sm_width * ((unpadded_bytes + sm_width - 1) / sm_width) + max_elt_size;
  return padded_bytes;
}

//! cuEmbed main namespace
namespace cuembed {

/*!
 * \brief Addresser calculates the input/output address for a specific lookup
 * sample based on the threadIdx and the number of threads per sample.
 *
 * For simplicity reasons, boundary checks are removed.
 *
 * Current assumptions:
 *   1. sample_id directly corresponds to the physical row in the embedding
 *      table. This may change with an index mapping or an embedding cache.
 *   2. Each thread performs a single load of VecT from an embedding row.
 *
 * \tparam InputT Type of embedding element, e.g. <tt>float</tt>, <tt>half</tt>.
 * \tparam OutputT Type of return element, e.g. <tt>float</tt>, <tt>half</tt>.
 * \tparam IndexT Integer type of index, e.g. <tt>int64_t</tt>.
 * \tparam VecT Vectorization type, e.g. <tt>float4</tt>.
 * \tparam mode Reduction mode in <tt>CombineMode</tt>.
 */
template <typename InputT,
          typename OutputT,
          typename IndexT,
          typename InputVecT,
          typename OutputVecT,
          CombineMode mode>
class Addresser {
 public:
  using InputType = InputT;
  using OutputType = OutputT;
  using InputVecType = InputVecT;
  using OutputVecType = OutputVecT;
  /*!
   * \brief Constructs Addresser.
   *
   * \param input Pointer to the beginning of the embedding table.
   * \param output Pointer to the output location.
   * \param sample_id The index of the sample within a batch (0 to sample_id-1).
   * \param num_hots Hotness of the embedding lookup workload.
   *
   * Example: For embedding_width 256, InputT/OutputT=float, VecT=float4, 64
   * threads (256/4) load an embedding row. Thus, threads_per_sample=64, and
   * thread_id_in_sample=threadIdx.x (0 to 63).
   */
  __device__ __host__ Addresser(const InputT* input,
                                OutputT* output,
                                const int64_t sample_id,
                                const int num_hots,
                                const int embed_width)
      : input_(input), embed_width_(embed_width) {
    int64_t output_offset = 0;
    if constexpr (mode == CombineMode::kSum || mode == CombineMode::kMean) {
      output_offset = sample_id * embed_width_;
    } else {
      output_offset = sample_id * num_hots * embed_width_;
    }
    output_ = output + output_offset;
  }
  /*!
   * \brief Get the vectorized address of the load for the specific lookup
   * index.
   *
   * \param index The embedding row index for this specific load.
   */
  __device__ __host__ __forceinline__ const InputVecT* GetEmbeddingAddress(
      const int64_t index) const {
    return reinterpret_cast<const InputVecT*>(input_ + index * embed_width_);
  }

  /*!
   * \brief Get the vectorized address of output for this sample_id.
   */
  __device__ __host__ __forceinline__ OutputVecT* GetOutputAddress() const {
    return reinterpret_cast<OutputVecT*>(output_);
  }

  /*!
   * \brief Get the the `i_hotness`th segment of the vectorized address of
   * output for this sample_id.

   * This assumes that the combine_mode is concatenation.
   */
  __device__ __host__ __forceinline__ OutputVecT* GetConcatOutputAddress(
      const int64_t i_hotness) const {
    return reinterpret_cast<OutputVecT*>(output_ + embed_width_ * i_hotness);
  }

 private:
  /// The pointer to the beginning of embedding table.
  const InputT* input_;
  /// The pointer to the location of output.
  OutputT* output_;
  /// Embedding row width.
  int embed_width_;
};

/*!
 * \brief Combiner takes the addresses calculated from the addresser and
 * read/write to the provided address.
 *
 * Depending on the combine_mode, it either accumulates the read internally with
 * a temporary vector, or writes out an output directly.
 *
 * We use template specialization for different behaviors of the combiner under
 * different reduction types.
 *
 * \tparam LoadVecT Vectorization type for loading, e.g. <tt>float4</tt>.
 * \tparam ReduceVecT Vectorization type for reduction, e.g., <tt>float4</tt>.
 * \tparam mode Reduction mode in <tt>CombineMode</tt>.
 */
template <typename InputVecT,
          typename ReduceVecT,
          typename OutputVecT,
          CombineMode combine_mode>
class Combiner {
 public:
  /*!
   * \brief Constructor of combiner.
   */
  __device__ __host__ Combiner() {}
  /*!
   * \brief When combine_mode is kConcat, write the gathered result to specified
   * output location.
   *
   * \param output Pointer of the output location.
   */
  __device__ __host__ __forceinline__ void OutputForConcatIfNeeded(
      OutputVecT* output) const {}
  /*!
   * \brief When combine_mode is kSum, write the gathered result to specified
   * output location.
   *
   * \param output Pointer of the output location.
   */
  __device__ __host__ __forceinline__ void OutputForReductionIfNeeded(
      OutputVecT* output) const {}
  /*!
   * \brief Reads the LoadVecT from a specified location, applies the weight,
   * and accumulates/overwrites the internal ReduceVecT depending on the reduce
   * type.
   *
   * \param input Pointer of the input location.
   * \param weight Weight to be applied to the input values.
   */
  template <typename ElemT>
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input,
                                                  const ElemT weight) {}
  /*!
   * \brief Reads the LoadVecT from a specified location and
   * accumulates/overwrites the internal vector depending on the reduce
   * type.
   *
   * \param input Pointer of the input location.
   */
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input) {}
};

/*!
 * \brief Template specialization for reduction type concatenation.
 *
 * The OutputForConcatIfNeeded() is a noop. The combiner reads from the input
 * and internally accumulates the read. It writes out the reduction result when
 * OutputForReductionIfNeeded is called.
 */
template <typename InputVecT, typename ReduceVecT, typename OutputVecT>
class Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kSum> {
 public:
  __device__ __host__ Combiner() { memset(&sum_, 0, sizeof(ReduceVecT)); }

  template <typename ElemT>
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input,
                                                  const ElemT weight) {
    // This level of indirection is required to load the vector in one
    // instruction.
    InputVecT tmp_vec = *input;
    sum_ += VecCast<ReduceVecT, InputVecT>(tmp_vec) * weight;
  }

  __device__ __host__ __forceinline__ void Gather(const InputVecT* input) {
    // This level of indirection is required to load the vector in one
    // instruction.
    InputVecT tmp_vec = *input;
    sum_ += tmp_vec;
  }

  __device__ __host__ __forceinline__ void OutputForReductionIfNeeded(
      OutputVecT* output) const {
    *output = VecCast<OutputVecT, ReduceVecT>(sum_);
  }

  __device__ __host__ __forceinline__ void OutputForConcatIfNeeded(
      OutputVecT* output) const {}

 protected:
  ReduceVecT sum_;
};

/*!
 * \brief Template specialization for reduction type mean.
 *
 * Everything except for output writing is the same as the specialization of
 * summation. Mean calculation is applied when outputting.
 */
template <typename InputVecT, typename ReduceVecT, typename OutputVecT>
class Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kMean>
    : public Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kSum> {
 public:
  template <typename ElemT>
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input,
                                                  const ElemT weight) {
    Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kSum>::Gather(
        input, weight);
    accumulated_weight_ += weight;
  }

  __device__ __host__ __forceinline__ void Gather(const InputVecT* input) {
    Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kSum>::Gather(
        input);
    accumulated_weight_ += 1.0f;
  }

  __device__ __host__ __forceinline__ void OutputForReductionIfNeeded(
      OutputVecT* output) const {
    if (accumulated_weight_ == 0.f) {
      memset(output, 0, sizeof(OutputVecT));
    } else {
      // This follows Tensorflow's combiner=='mean' implementation.
      // For `mean` calculation in PyTorch, weights_ is assumed to be nullptr.
      // Thus accumulated_weight_ equals to num_hots, which also gives the
      // correct output.
      *output = VecCast<OutputVecT, ReduceVecT>(this->sum_ *
                                                (1.0f / accumulated_weight_));
    }
  }

 private:
  float accumulated_weight_ = 0.f;
};

/*!
 * \brief Template specialization for reduction type concatenation.
 *
 * OutputForReductionIfNeeded() is a noop. The combiner reads from the input
 * and writes output when OutputForConcatIfNeeded is called.
 *
 * ReduceVecT does not affect the combiner since there is no math needed for
 * concat.
 */
template <typename InputVecT, typename ReduceVecT, typename OutputVecT>
class Combiner<InputVecT, ReduceVecT, OutputVecT, CombineMode::kConcat> {
 public:
  __device__ __host__ Combiner() {}
  template <typename ElemT>
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input,
                                                  const ElemT weight) {
    tmp_ = *input;
  }
  __device__ __host__ __forceinline__ void Gather(const InputVecT* input) {
    tmp_ = *input;
  }

  __device__ __host__ __forceinline__ void OutputForConcatIfNeeded(
      OutputVecT* output) const {
    *output = tmp_;
  }
  __device__ __host__ __forceinline__ void OutputForReductionIfNeeded(
      OutputVecT* output) const {}

 protected:
  InputVecT tmp_;
};

/*!
 * \brief IndexLoader is a helper class that returns the lookup indices of a
 * specific lookup sample.
 *
 * Different  index layout (CSR, fixed hotness) corresponds to a specific
 * template specialization of IndexLoader. The current implementation supports
 * two index layouts: CSR and fixed hotness. For fixed hotness, the offset type
 * is void, and hotness is specified via the num_hots variable. Fixed hotness
 * index loader loads all indices needed by the CTA into shared memory upon
 * initialization. For CSR indices, offsets array is assumed to be of size
 * batchsize + 1, corresponding to the offset data array for each sample.
 *
 *
 * \tparam IndexT Datatype for individual indices. e.g. <tt>int</tt>.
 * \tparam WeightT Datatype for per sample weights.
 * \tparam OffsetT CSR specialization of index layout. e.g., <tt>int</tt>. If
 * void, then fixed hotness.
 */
template <typename IndexT, typename WeightT, typename OffsetT>
class IndexLoader {
 public:
  // Type aliases
  using IndexType = IndexT;
  using WeightType = WeightT;
  using OffsetType = OffsetT;

  /*!
   * \brief Constructor of IndexLoader.
   *
   * \param batch_size Batch size of the lookup workload.
   * \param sample_id The sample this specific thread is working on.
   * \param indices The lookup indices.
   * \param offsets Offsets array of the CSR layout for indices.
   * \param num_hots Hotness for fixed hotness indices layout.
   */
  __device__ __host__ IndexLoader(const int batch_size,
                                  const int sample_id,
                                  const IndexT* indices,
                                  const WeightT* weights,
                                  const OffsetT* offsets,
                                  const int num_hots) {
    int sample_index_start = offsets[min(batch_size, sample_id)];
    int sample_index_stop = offsets[min(batch_size, sample_id + 1)];
    sample_hotness_ = sample_index_stop - sample_index_start;
    index_for_sample_ = const_cast<IndexT*>(indices) + sample_index_start;
    if (weights != nullptr) {
      weight_for_sample_ = const_cast<WeightT*>(weights) + sample_index_start;
    }
  }
  /*!
   * \brief Get the `i_hotness`th lookup index for the specific sample.
   */
  __device__ __host__ __forceinline__ IndexT
  GetLookUpIndex(const int i_hotness) const {
    return index_for_sample_[i_hotness];
  }
  /*!
   * \brief Get the corresponding weight for the specific sample.
   */
  __device__ __host__ __forceinline__ WeightT
  GetWeight(const int i_hotness) const {
    return ((weight_for_sample_ == nullptr) ? VecCast<WeightT, float>(1.0f)
                                            : weight_for_sample_[i_hotness]);
  }

  /*!
   * \brief Get the hotness for the specific sample.
   */
  __device__ __host__ __forceinline__ int GetHotness() const {
    return sample_hotness_;
  }

 private:
  /// Pointer to the lookup indices for this sample.
  IndexT* index_for_sample_ = nullptr;
  /// Pointer to the weight for each lookup index.
  WeightT* weight_for_sample_ = nullptr;
  /// Hotness for this sample.
  IndexT sample_hotness_;
};

/*!
 * \brief Template specialization for fixed hotness index loader.
 *
 * At construction, the index loader loads all lookup indices needed by the CTA
 * into shared memory. During GetLookupIndex it loads the index from shared
 * memory for faster lookup and reduced traffic to global memory.
 */
template <typename IndexT, typename WeightT>
class IndexLoader<IndexT, WeightT, void> {
 public:
  // Type aliases
  using IndexType = IndexT;
  using WeightType = WeightT;
  using OffsetType = void;
  __device__ __host__ IndexLoader(const int batch_size,
                                  const int sample_id,
                                  const IndexT* indices,
                                  const WeightT* weights,
                                  const void* offsets,
                                  const int num_hots)
      : hotness_(num_hots) {
#ifndef FOR_HOST_TEST  // Shared memory related code cannot be tested on host.
    int cta_index_start = blockIdx.x * blockDim.y * num_hots;
    int cta_index_stop =
        min(batch_size * num_hots, (blockIdx.x + 1) * blockDim.y * num_hots);
    LoadIndexToShmemAndSync(cta_index_start, cta_index_stop, indices, weights);
#endif
  }

  __device__ __host__ __forceinline__ IndexT
  GetLookUpIndex(const int i_hotness) const {
    return loaded_indices_[i_hotness + hotness_ * threadIdx.y];
  }

  __device__ __host__ __forceinline__ WeightT
  GetWeight(const int i_hotness) const {
    return ((loaded_weights_ == nullptr)
                ? VecCast<WeightT, float>(1.0f)
                : loaded_weights_[i_hotness + hotness_ * threadIdx.y]);
  }

  __device__ __host__ __forceinline__ int GetHotness() const {
    return hotness_;
  }

 private:
  /*!
   * \brief Loads lookup indices into shared memory.
   */
  __device__ void LoadIndexToShmemAndSync(const IndexT index_start,
                                          const IndexT index_stop,
                                          const IndexT* indices_,
                                          const WeightT* weights_) {
    extern __shared__ int32_t shmem_indices_raw[];
    loaded_indices_ = reinterpret_cast<IndexT*>(shmem_indices_raw);

    const int tid_in_cta = threadIdx.x + threadIdx.y * blockDim.x;
    const int cta_size = blockDim.x * blockDim.y;

    for (int i = index_start + tid_in_cta; i < index_stop; i += cta_size) {
      __pipeline_memcpy_async(
          &loaded_indices_[i - index_start], &indices_[i], sizeof(IndexT));
    }
    if (weights_ != nullptr) {
      loaded_weights_ = reinterpret_cast<WeightT*>(loaded_indices_ +
                                                   index_stop - index_start);
      if constexpr (std::is_same_v<WeightT, float>) {
        for (int i = index_start + tid_in_cta; i < index_stop; i += cta_size) {
          __pipeline_memcpy_async(
              &loaded_weights_[i - index_start], &weights_[i], sizeof(WeightT));
        }
      } else {
        // pipeline memcpy async does not support copy of single half. Also
        // faces address misalignment issue when copying multiple halves.
        // Falling back to regular lds instead.
        for (int i = index_start + tid_in_cta; i < index_stop; i += cta_size) {
          loaded_weights_[i - index_start] = weights_[i];
        }
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    __syncthreads();
  }

  /// Pointer to the loaded indices (in shmem).
  IndexT* loaded_indices_ = nullptr;
  WeightT* loaded_weights_ = nullptr;
  const IndexT hotness_;
};

// Helper functions for backward pass

template <typename IndexT, typename WeightT>
class GradIndexLoader {
 public:
  using IndexType = IndexT;
  using WeightType = WeightT;
  __device__ GradIndexLoader(const IndexT* __restrict__ transpose_indices,
                             const IndexT* __restrict__ transpose_sample_ids,
                             const WeightT* __restrict__ transpose_weights,
                             const int nnz,
                             const int nz_block_size) {
#ifndef FOR_HOST_TEST
    LoadIndexToShmemAndSync(transpose_indices,
                            transpose_sample_ids,
                            transpose_weights,
                            nnz,
                            nz_block_size);
#endif
  }

  __device__ __forceinline__ void LoadIndexToShmemAndSync(
      const IndexT* __restrict__ transpose_indices,
      const IndexT* __restrict__ transpose_sample_ids,
      const WeightT* __restrict__ transpose_weights,
      const int nnz,
      const int nz_block_size) {
    extern __shared__ int32_t shmem_raw[];
    const int bytes_per_ydim =
        GetSmemBytesPerNzBlock<IndexT, WeightT>(nz_block_size);
    int8_t* smem_raw_bytes = reinterpret_cast<int8_t*>(shmem_raw);
    sh_transpose_indices = reinterpret_cast<IndexT*>(
        &smem_raw_bytes[threadIdx.y * bytes_per_ydim]);
    sh_transpose_sample_ids =
        reinterpret_cast<IndexT*>(&sh_transpose_indices[nz_block_size]);
    sh_transpose_weights =
        reinterpret_cast<WeightT*>(&sh_transpose_sample_ids[nz_block_size]);
    should_write =
        reinterpret_cast<bool*>(&sh_transpose_weights[nz_block_size]);
    should_atomic = reinterpret_cast<bool*>(&should_write[nz_block_size]);

    const int nz_start =
        ((blockIdx.x * blockDim.y) + threadIdx.y) * nz_block_size;
    block_nnz = max(0, min(nz_start + nz_block_size, nnz) - nz_start);
    for (int i = threadIdx.x; i < block_nnz; i += blockDim.x) {
      sh_transpose_indices[i] = transpose_indices[nz_start + i];
      sh_transpose_sample_ids[i] = transpose_sample_ids[nz_start + i];

      // Only for weighted
      if (transpose_weights != nullptr) {
        sh_transpose_weights[i] = transpose_weights[nz_start + i];
      } else {
        sh_transpose_weights[i] = 1.;
      }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < block_nnz; i += blockDim.x) {
      bool first = sh_transpose_indices[i] == sh_transpose_indices[0];
      bool last = (i == block_nnz - 1);
      bool diff = (i < block_nnz)
                      ? sh_transpose_indices[i + 1] != sh_transpose_indices[i]
                      : true;
      should_write[i] = diff && !(first || last);
      should_atomic[i] = (diff && first) || last;
    }
    __syncthreads();
  }

  __device__ __forceinline__ IndexT GetIndex(const int i) {
    return sh_transpose_indices[i];
  }

  __device__ __forceinline__ IndexT GetSampleId(const int i) {
    return sh_transpose_sample_ids[i];
  }

  __device__ __forceinline__ WeightT GetWeight(const int i) {
    return sh_transpose_weights[i];
  }

  __device__ __forceinline__ bool ShouldWrite(const int i) {
    return should_write[i];
  }

  __device__ __forceinline__ bool ShouldAtomic(const int i) {
    return should_atomic[i];
  }

  __device__ __forceinline__ int GetBlockNnz() { return block_nnz; }

 private:
  IndexT* sh_transpose_indices;
  IndexT* sh_transpose_sample_ids;
  WeightT* sh_transpose_weights;
  bool* should_write;
  bool* should_atomic;
  int block_nnz;
};

template <typename GradT, typename GradVecT>
class GradAddresser {
 public:
  using GradType = GradT;
  using GradVecType = GradVecT;
  __device__ __host__ GradAddresser(const GradT* __restrict__ grad_y,
                                    GradT* __restrict__ grad_embedding,
                                    const int embed_width) {
    grad_y_ = grad_y;
    grad_embedding_ = grad_embedding;
    embed_width_ = embed_width;
  }

  __device__ __host__ __forceinline__ const GradVecT* GetGradResultAddress(
      const int col) {
    return reinterpret_cast<const GradVecT*>(grad_y_ + col * embed_width_);
  }

  __device__ __host__ __forceinline__ GradVecT* GetGradEmbeddingAddress(
      const int row) {
    return reinterpret_cast<GradVecT*>(grad_embedding_ + row * embed_width_);
  }

 private:
  const GradT* grad_y_;
  GradT* grad_embedding_;
  int embed_width_;
};

template <typename GradT, typename GradVecT, typename WeightT>
class GradCombiner {
 public:
  using GradType = GradT;
  using GradVecType = GradVecT;
  using WeightType = WeightT;
  __device__ __host__ GradCombiner() {
    memset(&accumulator, 0, sizeof(accumulator));
  }

  __device__ __host__ __forceinline__ void Gather(const GradVecT* input,
                                                  WeightT weight) {
    GradVecT tmp_vec = *input;
    accumulator += tmp_vec * weight;
  }

  __device__ __host__ __forceinline__ void Gather(const GradVecT* input) {
    GradVecT tmp_vec = *input;
    accumulator += tmp_vec;
  }

  __device__ __host__ __forceinline__ void WriteOrAtomic(GradVecT* output,
                                                         bool write_flag,
                                                         bool atomic_flag) {
    if (write_flag) {
      *output = accumulator;
      memset(&accumulator, 0, sizeof(accumulator));
    }
    if (atomic_flag) {
#ifdef FOR_HOST_TEST
      *output += accumulator;
#else
      VecAtomicAdd(output, accumulator);
#endif
      memset(&accumulator, 0, sizeof(accumulator));
    }
  }

 private:
  GradVecT accumulator;
};

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_EMBEDDING_LOOKUP_OPS_CUH_
