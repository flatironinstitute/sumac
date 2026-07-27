import math
import torch
import torch.nn.functional as F
import time
from pytest import mark
import numpy as np
from numpy.typing import NDArray
from typing import Any
from sumac import sumac_factorize
from sumac.data import dense_to_sparse
import scipy as sp

from sumac.config import SumacMethod, SumacConfig

def generate_low_rank_data(m=1000, n=1000, d=16, noise_level=0.01, density=0.1, seed=0):
    """
    Generate sparse non-negative matrix S = ReLU(A_true @ B_true.T - bias + noise).
    """
    torch.manual_seed(seed)
    # Generate factors from Normal(0, 1) to get both positive and negative values
    # The bias term will control the density after ReLU.
    A_true = torch.randn(m, d)
    B_true = torch.randn(n, d)
    
    # Compute product
    S_latent = A_true @ B_true.T
    
    # Set a bias based on the target density
    # i.e. find a value in the tensor for which m*n*density elements
    # will be greater, so we can subtract that, thus setting the others
    # to zero after the relu
    S_flat = S_latent.flatten()
    bias_idx = math.ceil(len(S_flat) * density)
    bias_val = torch.sort(S_flat, descending=True)[0][bias_idx]
    S_dense = S_latent - bias_val
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn(m, n) * noise_level * S_dense[S_dense > 0].std()
        S_dense = S_dense + noise
        
    # Ensure non-negativity
    S_dense = torch.relu(S_dense)
    
    # Convert to sparse
    S_index, S_value = dense_to_sparse(S_dense)
    return S_index, S_value, m, n


def torch2scipy_svd(S_index, S_value, n, k=16):
    rows = S_index[0].numpy()
    cols = S_index[1].numpy()
    data = S_value.numpy().astype(np.float64)
    S_coo = sp.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    S_coo.sum_duplicates() #remove dups
    S_sparse = S_coo.tocsr()

    u: NDArray[Any] #make type checker happy
    s: NDArray[Any]
    vt: NDArray[Any]
    
    u, s, vt = sp.sparse.linalg.svds(S_sparse, k=k)

    # sort in descending order, as svds may return results in random order
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    vt = vt[idx, :]
    s_sqrt_diag = np.diag(np.sqrt(s))
    A = u @ s_sqrt_diag
    B = vt.T @ s_sqrt_diag
    return torch.FloatTensor(A), torch.FloatTensor(B)


@mark.parametrize("method, num_blocks", [(SumacMethod.GD, 2), (SumacMethod.SALSA, 25)])
def test_low_rank(method: SumacMethod, num_blocks: int):
    m = 10000
    n = 10000
    d_true = 16
    d_fit = 16
    target_density = 0.01
    iters = 300

    # TODO: decide whether to actually keep the manually-call-the-test interface
    print(f"--- Low-Rank Ground Truth Test (m={m}, n={n}, d_true={d_true}, density={target_density}) ---")
    S_index, S_value, m, n = generate_low_rank_data(m, n, d_true, density=target_density, noise_level=0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    actual_density = len(S_value) / (m * n)
    print(f"Actual density: {actual_density:.4f} ({len(S_value)} non-zeros)")
    
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    
    config = SumacConfig(
        rank = d_fit,
        max_iterations = iters,
        method = method,
        learning_rate = 0.1,
        num_blocks = num_blocks,
        seed = 0
    )
    print(f"\n>> Running SUMAC with method: {method}")
    t_start = time.time()
    A, B, costs = sumac_factorize(
        S_index=S_index,
        S_value=S_value,
        shape=(m, n),
        config=config
    )
    elapsed = time.time() - t_start
    
    # Calculate final error
    with torch.no_grad():
        S_pred = torch.relu(A @ B.T)
        # Reconstruct dense S for checking
        eval_device = S_pred.device
        S_index_eval = S_index.to(eval_device)
        S_value_eval = S_value.to(eval_device)
        S_dense = torch.zeros(m, n, dtype=S_value_eval.dtype, device=eval_device)
        S_dense[S_index_eval[0], S_index_eval[1]] = S_value_eval
        mse = F.mse_loss(S_pred, S_dense).item()
    
    print(f"Method {method} completed in {elapsed:.2f}s. Final MSE: {mse:.6f}")
    # TODO: This is not a very strong assertion
    assert mse < 0.08, f"{method} final MSE {mse:.6f} is too large."

if __name__ == "__main__":
    test_low_rank(SumacMethod.GD, 2)
    test_low_rank(SumacMethod.SALSA, 25)
