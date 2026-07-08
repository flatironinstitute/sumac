import torch

from .jit_kernel import *

torch.compile(mode='max-autotune')
def relu_block(A, B):
    S = torch.relu(A @ B.T)
    ref_sum = S.sum()
    ref_sum2 = (S * S).sum()
    return ref_sum, ref_sum2

def test_relu_abt_reduce_fused_consistency(
    device: str = "cuda",
    M: int = 1391,
    N: int = 138955,
    D: int = 16,
    BK: int = 32,
    MS: int = 4,
    BM: int = 128,
    seed: int = 0,
):
    assert torch.cuda.is_available()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    A = torch.randn(M, D, device=device, dtype=torch.float32, generator=g)
    B = torch.randn(N, D, device=device, dtype=torch.float32, generator=g)

    # Warm up JIT compile
    got_sum, got_sum2 = relu_abt_reduce_fused(A, B, BK=BK, MS=MS, BM=BM)
    ref_sum, ref_sum2 = relu_block(A,B)
    torch.cuda.synchronize()

    # Reference
#    S = torch.relu(A @ B.T)
#    ref_sum = S.sum(dtype=torch.float64)
#    ref_sum2 = (S * S).sum(dtype=torch.float64)
    with torch.cuda.nvtx.range("reference"):
        ref_sum, ref_sum2 = relu_block(A,B)
    # Kernel result
    with torch.cuda.nvtx.range("custom kernel"):
        got_sum, got_sum2 = relu_abt_reduce_fused(A, B, BK=BK, MS=MS, BM=BM)
    torch.cuda.synchronize()

    got_sum = got_sum.squeeze()
    got_sum2 = got_sum2.squeeze()

    print(f"ref_sum  = {ref_sum.item():.12e}")
    print(f"got_sum  = {got_sum.item():.12e}")
    print(f"rel err  = {((got_sum - ref_sum).abs() / ref_sum.abs().clamp_min(1e-30)).item():.12e}")
    print()
    print(f"ref_sum2 = {ref_sum2.item():.12e}")
    print(f"got_sum2 = {got_sum2.item():.12e}")
    print(f"rel err  = {((got_sum2 - ref_sum2).abs() / ref_sum2.abs().clamp_min(1e-30)).item():.12e}")


if __name__ == "__main__":
    test_relu_abt_reduce_fused_consistency(D=16)
    print()
    test_relu_abt_reduce_fused_consistency(D=13)
