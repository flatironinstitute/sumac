import torch
from relu_batc_jit.api import relu_bat_c_fused
from relu_bat_reduce_jit.api import relu_bat_reduce_fused
import pytest 

@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_relu_batc_kernel(D: int):
    N = 1458
    M = 145408
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    BM = 256
    BK = 32
    if D == 16:
        MS = 4
    elif D == 32:
        MS = 2
    else:
        MS = 1

    result = relu_bat_c_fused(A, B, C, BM=BM, BK=BK, MS=MS)
    reference = torch.relu(B @ A.T) @ C

    torch.cuda.synchronize()
    torch.testing.assert_close(result, reference)

@pytest.mark.parametrize("D", [13, 16, 17, 32, 64, 128, 256])
def test_relu_bat_reduce_kernel(D: int):
    N = 1458
    M = 145408
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)

    BM = 256
    BK = 32
    MS = 1

    result_1, result_2 = relu_bat_reduce_fused(B, A, BM=BM, BK=BK, MS=MS)
    tmp = torch.relu(B @ A.T)
    reference_1 = tmp.sum()
    reference_2 = (tmp * tmp).sum()
    torch.cuda.synchronize()

    torch.testing.assert_close(result_1.squeeze(), reference_1)
    torch.testing.assert_close(result_2.squeeze(), reference_2)

