import numpy as np
import torch
import math
from typing import Tuple

from _sumac.train_als import refactor

def _init_factors_salsa(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
    gen: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct translation of salsa_init.m logic.
    - Initialized with A>0, B<0.
    - Analytically rescaled so initial cost is approximately 2.
    """
    device = S_value.device
    dtype = S_value.dtype

    # Init factor before scaling, so that A@B is all negative
    S = torch.sparse_coo_tensor(
        S_index.to(device=device, dtype=torch.long), S_value,
        size=(m, n), device=device, dtype=dtype).coalesce()  # important for possible duplicate entries
    R_A = torch.rand((n, d), device=device, dtype=dtype, generator=gen) #dense random matrix in U(0,1)
    R_B = torch.rand((m, d), device=device, dtype=dtype, generator=gen) #dense random matrix in U(0,1)
    A = torch.sqrt(torch.sparse.mm(S, R_A))
    B = -torch.sqrt(torch.sparse.mm(S.T, R_B))

    # Optimize scale (alpha=scale^2) to set initial cost = 2 on non-zero entries
    # Goal: ||Sij - alpha*Lij|| / ||Sij|| = 2
    # Result (square both sides and cancel): alpha^2*||Lij||^2 - 2*alpha*(Sij.Lij) - 3*||Sij||^2 = 0
    # Lij is the dot product of factor rows at observed indices Sij
    Lij = torch.sum(A[S_index[0], :] * B[S_index[1], :], dim=1)
    
    ssqS = torch.sum(S_value ** 2)
    ssqL = torch.sum(Lij ** 2)
    SdotL = torch.sum(S_value * Lij)

    a = ssqL
    b = -2.0 * SdotL
    c = -3.0 * ssqS

    # Quadratic formula for alpha = scale^2
    discriminant = b**2 - 4 * a * c
    alpha = (-b + torch.sqrt(discriminant)) / (2 * a)
    scale = torch.sqrt(alpha)

    # Apply scale and clamp to [0, 1] as effectively done by fminbnd(0,1)
    scale = torch.clamp(scale, 0.0, 1.0)
    A = scale * A
    B = scale * B

    # 4) Refactor
    A, B = refactor(A, B)
    return A, B

def _init_factors_testcase(
    m: int, d: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Init (A,B) deterministically for test purpose
    """
    np.random.seed(0)
    A = torch.FloatTensor(np.random.random((m, d)))
    B = -A
    return A, B

def _init_factors_factor_specific(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factor-specific init based on row/col marginals of S.
    Produces A (m,d), B (n,d), then refactor().
    """
    # column sums
    sum1 = torch.zeros(n, dtype=S_value.dtype, device=S_value.device)
    sum1.index_add_(0, S_index[1], S_value)
    sum1 = sum1.reshape(-1, 1)  # (n,1)

    # row sums
    sum2 = torch.zeros(m, dtype=S_value.dtype, device=S_value.device)
    sum2.index_add_(0, S_index[0], S_value)
    sum2 = sum2.reshape(-1, 1)  # (m,1)

    # global sum
    sumS = S_value.sum()
    scaleA = torch.sqrt(n * sum2 / sumS)   # (m,1)
    scaleB = torch.sqrt(m * sum1 / sumS)   # (n,1)
    A = scaleA * (1 + torch.rand((m, d), dtype=torch.float32, device=S_value.device) / d) / 2
    B = -scaleB * (1 + torch.rand((n, d), dtype=torch.float32, device=S_value.device) / d) / 2
    A, B = refactor(A, B)
    return A, B


def _init_factors_factor_agnostic(
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factor-agnostic init with your improved scaling (keeps your exact math).
    Produces A (m,d), B (n,d), then refactor().
    """
    scale = 0.5 * math.sqrt(S_value.mean() / d)
    A = (torch.rand((m, d), device=S_value.device) * scale)
    B = (-torch.rand((n, d), device=S_value.device) * scale)
    A, B = refactor(A, B)
    return A, B


def als_init_factors(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
    opts: dict,
    test_flag: bool,
    gen: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single entry point for init. Chooses:
      - testcase init if test_flag=True
      - factor-specific init if opts['factor_init']=True
      - factor-agnostic init otherwise
    """
    if opts.get('method', '').upper() == 'SALSA':
        print('salsa init...')
        return _init_factors_salsa(S_index, S_value, m, n, d, gen)
    if test_flag:
        return _init_factors_testcase(m, d)
    if opts.get("factor_init", False):
        return _init_factors_factor_specific(S_index, S_value, m, n, d)
    else:
        return _init_factors_factor_agnostic(S_value, m, n, d)

def als_post_process_factors(
    rmse: float,
    it: int,
    rmse_hist: list,
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1) if rmse > 1.0: shrink weights, refactor, reset momentum buffers
     2) if it > 1 and rmse > rmse_hist[-2]: reset momentum buffers
    """
    if rmse > 1.0:
        A = 0.5 * A
        B = 0.5 * B
        A, B = refactor(A, B)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
    if it > 1 and rmse > rmse_hist[-2]:
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)

    return A, B, dA, dB

def als_early_stop(
    it: int,
    elapsed: float,
    jacc: float,
    jacc_hist: list,
    opts: dict,
) -> bool:
    """
    Early stopping criteria, identical to your existing logic.
    """
    if elapsed > opts["time_limit"]:
        return True
    if jacc < opts["tol_abs"]:
        return True
    if it >= 2 * opts["tol_window"]:
        w = opts["tol_window"]
        m1 = sum(jacc_hist[-w:]) / w
        m2 = sum(jacc_hist[-2 * w : -w]) / w
        if (m2 - m1) / m2 < opts["tol_rel"]:
            return True
    return False
