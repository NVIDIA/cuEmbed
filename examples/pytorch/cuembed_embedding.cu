// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "cuembed/include/embedding_lookup.cuh"
#include "cuembed/include/index_transforms.cuh"

torch::Tensor cuembed_embedding_forward(const torch::Tensor params,
                                        const torch::Tensor indices,
                                        const torch::Tensor offsets,
                                        const torch::Tensor weights,
                                        const std::string mode) {
  AT_ASSERT(indices.is_cuda());
  AT_ASSERT(offsets.is_cuda());
  AT_ASSERT(params.is_cuda());
  AT_ASSERT(params.is_contiguous());
  AT_ASSERT(params.scalar_type() == torch::ScalarType::Float);
  if (weights.defined()) {
    AT_ASSERT(weights.scalar_type() == torch::ScalarType::Float);
  }
  AT_ASSERT(indices.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(offsets.scalar_type() == torch::ScalarType::Long);
  using IndexType = int64_t;

  int num_features = params.size(0);
  int embed_width = params.size(1);

  int batch_size = offsets.numel() - 1;
  auto outputs = torch::empty(batch_size * embed_width, params.options());

  AT_ASSERT(mode == "sum");
  auto combine_mode = cuembed::CombineMode::kSum;
  cuembed::EmbeddingForward(
      params.data_ptr<float>(),
      embed_width,
      indices.contiguous().data_ptr<IndexType>(),
      offsets.contiguous().data_ptr<IndexType>(),
      weights.defined() ? weights.contiguous().data_ptr<float>() : nullptr,
      batch_size,
      0,
      combine_mode,
      outputs.mutable_data_ptr<float>(),
      at::cuda::getCurrentCUDAStream());

  return outputs.reshape({batch_size, embed_width});
}

torch::Tensor cuembed_extract_row_ids_from_csr(const torch::Tensor offsets,
                                               const int64_t nnz) {
  AT_ASSERT(offsets.is_cuda());
  AT_ASSERT(offsets.scalar_type() == torch::ScalarType::Long);
  using IndexType = int64_t;

  int batch_size = offsets.size(0);
  auto row_ids = torch::empty(nnz, offsets.options());
  cuembed::ExtractRowIdsFromCSR(offsets.data_ptr<IndexType>(),
                                batch_size,
                                row_ids.mutable_data_ptr<IndexType>(),
                                at::cuda::getCurrentCUDAStream());
  return row_ids;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuembed_transpose(
    const torch::Tensor rows,
    const torch::Tensor cols,
    const torch::Tensor weights) {
  AT_ASSERT(rows.is_cuda());
  AT_ASSERT(cols.is_cuda());
  AT_ASSERT(rows.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(cols.scalar_type() == torch::ScalarType::Long);
  if (weights.defined()) {
    AT_ASSERT(weights.scalar_type() == torch::ScalarType::Float);
  }

  int nnz = rows.size(0);
  using IndexType = int64_t;
  auto transpose_rows = torch::empty(nnz, rows.options());
  auto transpose_cols = torch::empty(nnz, cols.options());

  // TODO(niskos): Propely return a None tensor if there are no weights
  auto transpose_weights_size = weights.defined() ? nnz : 0;
  auto transpose_weights = torch::empty(
      transpose_weights_size, at::device(rows.device()).dtype(at::kFloat));

  size_t lwork = 0;
  cuembed::Transpose<IndexType, float>(
      rows.data_ptr<IndexType>(),
      cols.data_ptr<IndexType>(),
      weights.defined() ? weights.contiguous().data_ptr<float>() : nullptr,
      nnz,
      transpose_rows.mutable_data_ptr<IndexType>(),
      transpose_cols.mutable_data_ptr<IndexType>(),
      transpose_weights.mutable_data_ptr<float>(),
      nullptr,
      &lwork,
      at::cuda::getCurrentCUDAStream());
  auto work = torch::empty(lwork, at::device(rows.device()).dtype(at::kByte));
  cuembed::Transpose<IndexType, float>(
      rows.data_ptr<IndexType>(),
      cols.data_ptr<IndexType>(),
      weights.defined() ? weights.contiguous().data_ptr<float>() : nullptr,
      nnz,
      transpose_rows.mutable_data_ptr<IndexType>(),
      transpose_cols.mutable_data_ptr<IndexType>(),
      transpose_weights.mutable_data_ptr<float>(),
      reinterpret_cast<char*>(work.mutable_data_ptr<uint8_t>()),
      &lwork,
      at::cuda::getCurrentCUDAStream());
  return {transpose_rows, transpose_cols, transpose_weights};
}

torch::Tensor cuembed_embedding_backward(
    const torch::Tensor y_grad,
    const int64_t num_categories,
    const torch::Tensor transpose_indices,
    const torch::Tensor transpose_sample_ids,
    const torch::Tensor transpose_weights) {
  AT_ASSERT(transpose_indices.is_cuda());
  AT_ASSERT(transpose_sample_ids.is_cuda());
  AT_ASSERT(y_grad.is_cuda());
  AT_ASSERT(y_grad.is_contiguous());

  AT_ASSERT(transpose_indices.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(transpose_sample_ids.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(y_grad.scalar_type() == torch::ScalarType::Float);
  using IndexType = int64_t;

  // Allocate grad_embedding
  int embed_width = y_grad.size(1);
  int nnz = transpose_indices.size(0);
  auto grad_embedding =
      torch::zeros(num_categories * embed_width, y_grad.options());

  // Call backward
  cuembed::EmbeddingBackward<float, IndexType>(
      y_grad.data_ptr<float>(),
      embed_width,
      num_categories,
      nnz,
      transpose_indices.contiguous().data_ptr<IndexType>(),
      transpose_sample_ids.contiguous().data_ptr<IndexType>(),
      nullptr, /*transpose_remapped_indices*/
      transpose_weights.defined()
          ? transpose_weights.contiguous().data_ptr<float>()
          : nullptr,
      true, /*skip_grad_init*/
      grad_embedding.mutable_data_ptr<float>(),
      nullptr, /*inverse_mapping*/
      at::cuda::getCurrentCUDAStream());

  return grad_embedding.reshape({num_categories, embed_width});
}

TORCH_LIBRARY(cuembed_pyt, m) {
  m.def(
      "cuembed_extract_row_ids_from_csr(Tensor offsets, int nnz)"
      " ->Tensor",
      &cuembed_extract_row_ids_from_csr);
  m.def(
      "cuembed_transpose(Tensor rows, Tensor cols, Tensor weights) ->"
      " (Tensor, Tensor, Tensor)",
      &cuembed_transpose);
  m.def(
      "cuembed_embedding_forward(Tensor params, Tensor indices,"
      " Tensor offsets, Tensor weights, str mode) -> Tensor",
      &cuembed_embedding_forward);
  m.def(
      "cuembed_embedding_backward(Tensor y_grad, int num_categories,"
      " Tensor transpose_indices, Tensor transpose_sample_ids, Tensor "
      "transpose_weights) -> Tensor",
      &cuembed_embedding_backward);
}
