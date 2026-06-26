import torch

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

class _CuEmbEmbedding(torch.autograd.Function):
  @staticmethod
  def forward(ctx, params, idx, offsets, weights=None):
    ctx.save_for_backward(idx, offsets, weights)
    ctx.num_categories = params.size(0)
    return cuembed_forward(params, idx, offsets, weights)

  @staticmethod
  def backward(ctx, out_grad):
    return cuembed_backward(ctx, out_grad)

def cuemb_embedding(params, idx, offsets, weights=None):
  if not torch.is_grad_enabled() or not params.requires_grad:
    return cuembed_forward(params, idx, offsets, weights)
  return _CuEmbEmbedding.apply(params, idx, offsets, weights)

# Fake registrations let torch.compile run shape propagation for the custom
# torch.ops calls without executing the CUDA kernels.
@torch.library.register_fake("cuembed_pyt::cuembed_extract_row_ids_from_csr")
def _(offsets, nnz):
  return torch.empty((nnz,), device=offsets.device, dtype=offsets.dtype)

@torch.library.register_fake("cuembed_pyt::cuembed_transpose")
def _(rows, cols, weights=None):
  transpose_rows = torch.empty_like(rows)
  transpose_cols = torch.empty_like(cols)
  transpose_weights_size = 0 if weights is None else cols.shape[0]
  transpose_weights = torch.empty(
      (transpose_weights_size,), device=rows.device, dtype=torch.float32)
  return transpose_rows, transpose_cols, transpose_weights

@torch.library.register_fake("cuembed_pyt::cuembed_embedding_forward")
def _(params, idx, offsets, weights=None, mode="sum"):
  batch_size = offsets.shape[0] - 1
  embedding_dim = params.shape[1]
  return torch.empty((batch_size, embedding_dim), device=params.device, dtype=params.dtype)

@torch.library.register_fake("cuembed_pyt::cuembed_embedding_backward")
def _(y_grad, num_categories, transpose_indices, transpose_sample_ids, transpose_weights=None):
  embed_width = y_grad.shape[1]
  return torch.empty((num_categories, embed_width), device=y_grad.device, dtype=y_grad.dtype)
