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

# Run compile test
test_cuembed_compile()
