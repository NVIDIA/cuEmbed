import torch
from torch import nn
from cuembed_pyt import cuemb_embedding

torch.manual_seed(0)

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