import torch
from _sumac.tuning import *
#from jit_kernel import relu_bat_c_fused

from api import relu_bat_c_fused

N = 1408
M = 145408
D = 16
A = torch.randn(N, D, device="cuda", dtype=torch.float32)
B = torch.randn(M, D, device="cuda", dtype=torch.float32)
C = torch.randn(N, D, device="cuda", dtype=torch.float32)

BM = 256
BK = 32
MS = 4

def make_step(BM: int, BK: int, MS: int):
    def step(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return relu_bat_c_fused(A, B, C, BM=BM, BK=BK, MS=MS)
    return step

step = make_step(BM=BM, BK=BK, MS=MS)

compiled_step = torch.compile(
    step,
    fullgraph=True,
    mode="reduce-overhead",
)

# def relu_bat_cuda_launcher():
#     tune_config = {
#         "BM": [32, 64],
#         "BK": [16, 32],
#         "num_ms": [1, 2, 4],
#     }

#     @autotune_cuda_kernel(
#     configs=tune_config,
#     key_fn=relu_bat_c_key,
#     constraint_fn=relu_bat_c_constraints,
#     validate_fn=relu_bat_c_validate,
#     cache_path="relu_bat_c_jit_autotune.json",
#     n_trials=1000,
#     warmup=5,
#     rep=50,
#     sampler=optuna.samplers.GridSampler(search_space=tune_config)        
#     )
#     def relu_batc(
#         A: torch.Tensor,
#         B: torch.Tensor,
#         C: torch.Tensor,
#         BM: int,
#         BK: int,
#         num_ms: int,
#     ) -> torch.Tensor:
#         return relu_bat_c_fused(A, B, C, BK=BK, MS=num_ms, BM=BM)
#     return relu_batc

# relu_batc_tuned = relu_bat_cuda_launcher()

for i in range(10):
    with torch.cuda.nvtx.range("Kernel launch"):
        Y = compiled_step(A, B, C)

    #torch.cuda.synchronize()

    print(Y.shape)

    ref = torch.relu(B @ A.T) @ C

    err = (Y - ref).abs().max().item()
    print(err)