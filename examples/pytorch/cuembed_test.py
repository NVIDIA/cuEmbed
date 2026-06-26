import torch
from torch import nn
from cuembed_pyt import cuemb_embedding

torch.manual_seed(0)

class CuEmbedModule(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, indices, offsets):
        return cuemb_embedding(self.weight, indices, offsets)

def test_cuembed(embedding_bag, indices, offsets, weights):
    if(weights != None):
        res = cuemb_embedding(embedding_bag.weight, indices, offsets, weights)
        ref = embedding_bag(indices, offsets, weights)
    else:
        res = cuemb_embedding(embedding_bag.weight, indices, offsets)
        ref = embedding_bag(indices, offsets)

    print('fprop test pass = ', (res == ref).all())

    embedding_bag.weight.grad = None
    torch.mean(res).backward()
    grad_res = embedding_bag.weight.grad.clone()

    embedding_bag.weight.grad = None
    torch.mean(ref).backward()
    grad_ref = embedding_bag.weight.grad.clone()

    # might not be exactly equal because cuEmbed uses atomics in back pass
    print('bprop test pass = ', torch.allclose(grad_res, grad_ref), '\n')

def test_cuembed_inference(embedding_bag, indices, offsets):
    # Exercises the no-autograd fast path added with the autograd.Function
    # interface: cuemb_embedding skips _CuEmbEmbedding.apply when grad is
    # disabled or the params don't require grad, calling the op directly.
    ref = embedding_bag(indices, offsets)

    with torch.no_grad():
        res_nograd = cuemb_embedding(embedding_bag.weight, indices, offsets)
    # frozen weights: grad still enabled, but params.requires_grad is False
    res_frozen = cuemb_embedding(embedding_bag.weight.detach(), indices, offsets)

    print('inference no_grad test pass = ', torch.allclose(res_nograd, ref))
    print('inference frozen-weight test pass = ', torch.allclose(res_frozen, ref))
    # fast path must not build an autograd graph
    print('inference no-grad-graph test pass = ', not res_nograd.requires_grad, '\n')

def test_cuembed_noncontiguous(embedding_bag, indices, offsets):
    # non-contiguous inputs exercise the .contiguous() handling in fwd/bwd
    weight = embedding_bag.weight
    d = weight.shape[1]
    w_nc = torch.cat([weight, weight], dim=1).detach()[:, :d]
    idx_nc = torch.stack([indices, indices], dim=1).reshape(-1)[::2]
    assert not w_nc.is_contiguous() and not idx_nc.is_contiguous()

    ref = embedding_bag(indices, offsets)
    with torch.no_grad():
        res = cuemb_embedding(w_nc, idx_nc, offsets)
    print('non-contiguous fprop test pass = ', torch.allclose(res, ref))

    grad_mask = torch.ones(ref.shape[0], 2 * d, device=ref.device)[:, ::2]
    assert not grad_mask.is_contiguous()
    weight.grad = None
    (cuemb_embedding(weight, idx_nc, offsets) * grad_mask).sum().backward()
    grad_res = weight.grad.clone()
    weight.grad = None
    (embedding_bag(indices, offsets) * grad_mask).sum().backward()
    grad_ref = weight.grad.clone()
    print('non-contiguous bprop test pass = ', torch.allclose(grad_res, grad_ref), '\n')

def test_cuembed_compile():
    print("\nTesting cuembed with torch.compile...")

    # Create a simple embedding bag
    k = 15  # num_embeddings
    d = 2   # embedding_dim
    batch_size = 256

    # Create indices and offsets
    indices = torch.randint(0, k, (batch_size,), device='cuda', dtype=torch.long)
    offsets = torch.arange(0, batch_size + 1, device='cuda', dtype=torch.long)

    # Create embedding bag
    embedding_bag = nn.EmbeddingBag(
        num_embeddings=k,
        embedding_dim=d,
        mode='sum',
        include_last_offset=True,
        padding_idx=None,
        dtype=torch.float32
    ).to(device='cuda')

    # Create and compile the module
    cuembed_module = CuEmbedModule(embedding_bag.weight)
    compiled_module = torch.compile(cuembed_module)

    # Test forward pass with no_grad
    with torch.no_grad():
        res = compiled_module(indices, offsets)
        ref = embedding_bag(indices, offsets)

    # Verify shapes and values
    print(f"Expected shape: [{batch_size}, {d}]")
    print(f"Actual shape: {res.shape}")
    print(f"Shape test pass: {res.shape == ref.shape}")
    print(f"Value test pass: {torch.allclose(res, ref)}")

def test_cuembed_compile_backward():
    print("\nTesting cuembed with torch.compile (training/backward)...")
    k, d, n = 958, 16, 4096
    eb = nn.EmbeddingBag(k, d, mode='sum', include_last_offset=True,
                         dtype=torch.float32).to('cuda')
    indices = torch.randint(0, k, (n,), device='cuda', dtype=torch.long)
    offsets = torch.arange(0, n + 1, device='cuda', dtype=torch.long)

    def run(weight):
        return cuemb_embedding(weight, indices, offsets)

    w_ref = eb.weight.detach().clone().requires_grad_(True)
    run(w_ref).sum().backward()
    g_ref = w_ref.grad.clone()

    w_c = eb.weight.detach().clone().requires_grad_(True)
    torch.compile(run)(w_c).sum().backward()
    g_c = w_c.grad.clone()

    print(f"compile backward grad test pass: {torch.allclose(g_ref, g_c, atol=1e-4)}")

# test cases
k = 958
embedding_bag = nn.EmbeddingBag(
                num_embeddings=k,
                embedding_dim=128,
                mode='sum',
                include_last_offset=True,  # type: ignore Argument of type "bool | None" cannot be assigned ...
                padding_idx=None,
                dtype=torch.float32
            ).to(device='cuda')

n = 2880000
indices = k * torch.rand([n], device='cuda')
indices = indices.to(dtype=torch.long)

offsets = torch.tensor([ i for i in range(n)]+[n],device='cuda')
weights = torch.rand([n], device='cuda', dtype=torch.float32)

test_cuembed(embedding_bag, indices, offsets, weights)
test_cuembed(embedding_bag, indices, offsets, None)

k = 2048
embedding_bag = nn.EmbeddingBag(
                num_embeddings=k,
                embedding_dim=64,
                mode='sum',
                include_last_offset=True,  # type: ignore Argument of type "bool | None" cannot be assigned ...
                padding_idx=None,
                dtype=torch.float32
            ).to(device='cuda')

n = 104217
indices = k * torch.rand([n], device='cuda')
indices = indices.to(dtype=torch.long)

offsets = torch.tensor([ i for i in range(n)]+[n],device='cuda', dtype=torch.long)
weights = torch.rand([n], device='cuda', dtype=torch.float32)

test_cuembed(embedding_bag, indices, offsets, weights)
test_cuembed(embedding_bag, indices, offsets, None)

# Run inference fast-path test
test_cuembed_inference(embedding_bag, indices, offsets)

# Run non-contiguous input test
test_cuembed_noncontiguous(embedding_bag, indices, offsets)

# Run compile tests
test_cuembed_compile()
test_cuembed_compile_backward()
