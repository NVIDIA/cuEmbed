from absl import app
from absl import flags

import torch
from torch.nn import functional as F

torch.ops.load_library("../../build/examples/pytorch/libcuembed_pyt.so")

cuembed_extract_row_ids_from_csr = (
    torch.ops.cuembed_pyt.cuembed_extract_row_ids_from_csr
)
cuembed_transpose = torch.ops.cuembed_pyt.cuembed_transpose
cuembed_embedding_forward = torch.ops.cuembed_pyt.cuembed_embedding_forward
cuembed_embedding_backward = torch.ops.cuembed_pyt.cuembed_embedding_backward

def cuembed_forward(params, idx, offsets, weights):
  return cuembed_embedding_forward(params, idx, offsets, weights, mode="sum")

def cuembed_backward(ctx, out_grad):
  idx = ctx.saved_tensors[0]
  offsets = ctx.saved_tensors[1]
  weights = ctx.saved_tensors[2]
  num_categories = ctx.num_categories
  nnz = idx.size(0)

  # Assuming equivalent of nn.EmbeddingBag's `include_last_offset=True`
  sample_ids = cuembed_extract_row_ids_from_csr(offsets[:-1], nnz)
  transpose_indices, transpose_sample_ids, transpose_weights = \
    cuembed_transpose(sample_ids, idx, weights)
  
  # This means weights = None during forward
  if(transpose_weights.numel() == 0):
      transpose_weights = None
  
  grad_embedding = cuembed_embedding_backward(
      out_grad, num_categories, transpose_indices, transpose_sample_ids, transpose_weights)

  # no grad for indices, offsets, or weights
  return grad_embedding, None, None, None

def setup_context(ctx, inputs, output):
  params, idx, offsets, weights = inputs
  ctx.save_for_backward(idx, offsets, weights)
  ctx.num_categories = params.size(0)

# Need to register this as a custom op to allow torch.compile to work
@torch.library.custom_op("cuemb::cuemb_embedding", mutates_args=())
def cuemb_embedding(
  params : torch.Tensor, idx : torch.Tensor, offsets : torch.Tensor, weights : torch.Tensor = None) -> torch.Tensor:
  return cuembed_forward(params, idx, offsets, weights)

@cuemb_embedding.register_fake
def _(params : torch.Tensor, idx : torch.Tensor, offsets : torch.Tensor, weights : torch.Tensor = None):
  batch_size = offsets.shape[0] - 1
  embedding_dim = params.shape[1]
  return torch.empty((batch_size, embedding_dim), device=params.device, dtype=params.dtype)


cuemb_embedding.register_autograd(cuembed_backward, setup_context=setup_context)
