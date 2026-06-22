import torch

from relu_batc_jit.jit_kernel import (
    relu_bat_c_fused_traced,
    make_trace_capture,
    export_trace_json as export_float_trace_json,
    relu_bat_c_fused,
)

def test_relu_batc_kernel_trace():
    D = 16
    N = 1408
    M = 1155072

    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    BM = 128
    BK = 128
    MS = 6

    blocks = (M + BM * MS - 1) // (BM * MS)
    num_groups = BM // 32

    trace = make_trace_capture(
        num_blocks=blocks,
        num_groups=num_groups,
        events_per_group=2560,
    )

    result = relu_bat_c_fused_traced(A, B, C, BM=BM, BK=BK, MS=MS, trace=trace)
    torch.cuda.synchronize()

    export_float_trace_json(trace, "trace_relu_batc_bk64.json")

    reference = torch.relu(B @ A.T) @ C
    torch.testing.assert_close(result, reference, rtol=10, atol=1e-3)

def test_relu_batc_kernel_ref():
    D = 16
    N = 1408
    M = 1155072

    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(M, D, device="cuda", dtype=torch.float32)
    C = torch.randn(N, D, device="cuda", dtype=torch.float32)

    BM = 128
    BK = 128
    MS = 6


    result = relu_bat_c_fused(A, B, C, BM=BM, BK=BK, MS=MS)
    torch.cuda.synchronize()

    reference = torch.relu(B @ A.T) @ C
    torch.testing.assert_close(result, reference, rtol=10, atol=1e-3)





if __name__ == "__main__":
    test_relu_batc_kernel_trace()
    # test_relu_batc_kernel_ref()
