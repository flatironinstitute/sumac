import torch
from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused
from sumac.kernels.relu_bat_reduce_jit.api import relu_bat_reduce_fused
import pytest 

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="custom CUDA kernel tests require CUDA",
)

FP32_MAX_ABS_ERROR = 5e-3
TF32_MAX_ABS_ERROR = 2.0


def relu_bat_c_tf32_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        return torch.relu(B @ A.T) @ C
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def assert_max_abs_close(
    result: torch.Tensor,
    reference: torch.Tensor,
    max_abs_error: float,
) -> None:
    error = (result - reference).abs().max().item()
    assert error <= max_abs_error, (
        f"max_abs_error={error:.6e} exceeds {max_abs_error:.6e}"
    )


@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_relu_batc_kernel(D: int):
    N = 2500
    M = 250000
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    BM = 128
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
    assert_max_abs_close(result, reference, FP32_MAX_ABS_ERROR)


@pytest.mark.parametrize("D", [16, 32, 64, 128, 256])
def test_relu_batc_tf32_mma_sync_kernel_max_abs(D: int):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("TF32 MMA sync kernel requires SM80 or newer")

    from sumac.kernels.relu_batc_tf32_jit.api import (
        relu_bat_c_tf32_mma_sync,
    )

    torch.manual_seed(D)
    N = 2500
    M = 250000
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    result = relu_bat_c_tf32_mma_sync(
        A,
        B,
        C,
        BM=128,
        BN=16,
        M_TILES=2,
        num_stages=1,
    )
    reference = relu_bat_c_tf32_reference(A, B, C)

    torch.cuda.synchronize()
    assert_max_abs_close(result, reference, TF32_MAX_ABS_ERROR)


@pytest.mark.parametrize(
    ("D", "params"),
    [
        (
            16,
            {
                "BM": 128,
                "BN": 64,
                "WGMMA_S_N": 64,
                "WGMMA_Y_N": 16,
                "num_stages": 2,
                "wgmma_mode": "RS",
            },
        ),
        (
            32,
            {
                "BM": 128,
                "BN": 64,
                "WGMMA_S_N": 64,
                "WGMMA_Y_N": 32,
                "num_stages": 2,
                "wgmma_mode": "RS",
            },
        ),
        (
            64,
            {
                "BM": 128,
                "BN": 64,
                "WGMMA_S_N": 64,
                "WGMMA_Y_N": 64,
                "num_stages": 2,
                "wgmma_mode": "RS",
            },
        ),
        (
            128,
            {
                "BM": 128,
                "BN": 64,
                "WGMMA_S_N": 64,
                "WGMMA_Y_N": 128,
                "num_stages": 2,
                "wgmma_mode": "RS",
            },
        ),
        (
            256,
            {
                "BM": 64,
                "BN": 16,
                "WGMMA_S_N": 16,
                "WGMMA_Y_N": 128,
                "num_stages": 1,
                "wgmma_mode": "SS",
            },
        ),
    ],
)
def test_relu_batc_tf32_wgmma_kernel_max_abs(D: int, params: dict):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("TF32 WGMMA kernel requires SM90")

    from sumac.kernels.relu_batc_tf32_jit.api import (
        relu_bat_c_tf32_wgmma,
    )

    torch.manual_seed(D)
    N = 2500
    M = 250000
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    result = relu_bat_c_tf32_wgmma(A, B, C, **params)
    reference = relu_bat_c_tf32_reference(A, B, C)

    torch.cuda.synchronize()
    assert_max_abs_close(result, reference, TF32_MAX_ABS_ERROR)


@pytest.mark.parametrize("D", [13, 16, 17, 32, 64, 128, 256])
def test_relu_bat_reduce_kernel(D: int):
    N = 2500
    M = 250000
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


@pytest.mark.parametrize("D", [13, 16, 32])
@pytest.mark.parametrize("include_sum_sr", [False, True])
def test_relu_bat_reduce_kernel_backward(D: int, include_sum_sr: bool):
    torch.manual_seed(D + int(include_sum_sr))
    M = 257
    N = 263
    A = torch.randn(M, D, device="cuda", dtype=torch.float32, requires_grad=True)
    B = torch.randn(N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)

    BM = 128
    BK = 32
    MS = 1

    sum_sr, sum_sr2 = relu_bat_reduce_fused(A, B, BM=BM, BK=BK, MS=MS)
    loss = 1.3 * sum_sr2.squeeze()
    if include_sum_sr:
        loss = loss + 0.7 * sum_sr.squeeze()
    loss.backward()

    tmp = torch.relu(A_ref @ B_ref.T)
    reference_loss = 1.3 * (tmp * tmp).sum()
    if include_sum_sr:
        reference_loss = reference_loss + 0.7 * tmp.sum()
    reference_loss.backward()

    torch.cuda.synchronize()
    assert_max_abs_close(A.grad, A_ref.grad, FP32_MAX_ABS_ERROR)
    assert_max_abs_close(B.grad, B_ref.grad, FP32_MAX_ABS_ERROR)
