import numpy as np
from tests.check_min_rank import find_large_positive_fullrank_square, find_large_positive_fullrank_square_multistart
from tests.test_sumac import generate_low_rank_data
import time
import argparse 
import scipy.io as sio
import scipy.sparse as sp

def _make_sparse(S_dense: np.ndarray):
    rows, cols = np.nonzero(S_dense)
    S_index = np.vstack([rows, cols]).astype(np.int64)
    S_value = S_dense[rows, cols]
    m, n = S_dense.shape
    return S_index, S_value, m, n


# ---------------------------------------------------------------------------
# Test 1a — Identity matrix: each row shares a positive column with NO other row,
#            so the largest all-positive submatrix is 1×1 → k == 1
# ---------------------------------------------------------------------------
def test_identity_k1():
    n = 5
    S_dense = np.eye(n, dtype=np.int64)
    S_index, S_value, m, n_ = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, m, n_,
                                               core_thresholds=(), n_seeds=50)
    assert best["k"] == 1, f"identity has no 2×2 all-positive submatrix, expected k=1, got {best['k']}"


# ---------------------------------------------------------------------------
# Test 1b — Diagonally-dominant all-positive matrix: every entry ≥ 1 and the
#            matrix is full rank → the algorithm should find k == n
# ---------------------------------------------------------------------------
def test_dense_full_rank():
    n = 5
    # n*I + J: all entries are 1 (off-diagonal) or n+1 (diagonal), full rank by DD
    S_dense = np.ones((n, n), dtype=np.int64) + n * np.eye(n, dtype=np.int64)
    S_index, S_value, m, n_ = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, m, n_,
                                               core_thresholds=(), n_seeds=50)
    assert best["k"] == n, f"expected k={n}, got {best['k']}"


# ---------------------------------------------------------------------------
# Test 2 — All-ones matrix: rank 1 → best k should be 1
# ---------------------------------------------------------------------------
def test_all_ones_rank1():
    m, n = 4, 5
    S_dense = np.ones((m, n), dtype=np.int64)
    S_index, S_value, mm, nn = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, mm, nn,
                                               core_thresholds=(), n_seeds=50)
    assert best["k"] == 1, f"expected k=1 (rank-1 matrix), got {best['k']}"


# ---------------------------------------------------------------------------
# Test 3 — Embedded 4×4 full-rank block in a 10×10 diagonal background.
#           The diagonally-dominant block guarantees full rank.
# ---------------------------------------------------------------------------
def test_embedded_block():
    block = np.array([
        [5, 1, 1, 1],
        [1, 5, 1, 1],
        [1, 1, 5, 1],
        [1, 1, 1, 5],
    ], dtype=np.int64)
    S_dense = np.diag(np.ones(10, dtype=np.int64))
    S_dense[:4, :4] = block
    S_index, S_value, m, n = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, m, n,
                                               core_thresholds=(2,), n_seeds=100)
    assert best["k"] >= 4, f"expected k>=4, got {best['k']}"


# ---------------------------------------------------------------------------
# Test 4 — Block-diagonal: 3×3 block and 5×5 block.
#           Algorithm should find the larger (5×5) block.
# ---------------------------------------------------------------------------
def test_block_diagonal_finds_larger():
    S_dense = np.zeros((8, 8), dtype=np.int64)
    # 3×3 diagonally-dominant block
    S_dense[:3, :3] = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]], dtype=np.int64)
    # 5×5 diagonally-dominant block
    b5 = 5 * np.eye(5, dtype=np.int64) + np.ones((5, 5), dtype=np.int64)
    S_dense[3:, 3:] = b5
    S_index, S_value, m, n = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, m, n,
                                               core_thresholds=(2,), n_seeds=100)
    assert best["k"] >= 5, f"expected k>=5 (5x5 block), got {best['k']}"


# ---------------------------------------------------------------------------
# Test 5 — Near-singular: all rows identical → rank 1
# ---------------------------------------------------------------------------
def test_repeated_rows_rank1():
    row = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    S_dense = np.tile(row, (8, 1))
    S_index, S_value, m, n = _make_sparse(S_dense)
    best = find_large_positive_fullrank_square(S_index, S_value, m, n,
                                               core_thresholds=(), n_seeds=50)
    assert best["k"] == 1, f"rank-1 matrix should give k=1, got {best['k']}"


# ---------------------------------------------------------------------------
# Test 6 — Float-valued matrix: values preserved without truncation.
#           Uses a 3×3 diagonally-dominant float matrix with entries in (0, 2).
# ---------------------------------------------------------------------------
def test_float_valued():
    S_dense_np = np.array([
        [1.5, 0.1, 0.2],
        [0.3, 1.8, 0.1],
        [0.2, 0.1, 1.6],
    ], dtype=np.float32)
    rows, cols = np.nonzero(S_dense_np)
    S_index = np.vstack([rows, cols]).astype(np.int64)
    S_value = S_dense_np[rows, cols]
    m, n = S_dense_np.shape
    best = find_large_positive_fullrank_square(S_index, S_value, m, n,
                                               core_thresholds=(), n_seeds=20)
    assert best["k"] == 3, (
        f"3×3 diagonally-dominant float matrix should give k=3, got {best['k']}. "
        "If k<3, float truncation bug may still be present."
    )


# ---------------------------------------------------------------------------
# Test 7 — Large sparse matrix (smoke / regression test).
# S = ReLU(A @ B.T - bias) is NOT rank-d after the nonlinear ReLU step,
# so we only assert the algorithm completes and finds a nontrivial submatrix.
# ---------------------------------------------------------------------------
def test_large_sparse_smoke():
    S_index_t, S_value_t, m, n = generate_low_rank_data(
        1000, 1000, d=16, density=0.1, noise_level=0.0, seed=0
    )
    S_index = S_index_t.numpy().astype(np.int64)
    S_value = S_value_t.numpy()
    best = find_large_positive_fullrank_square(
        S_index, S_value, m=m, n=n,
        core_thresholds=(5,),
        n_seeds=200,
        rng_seed=123,
    )
    assert best["k"] > 0, f"algorithm should find at least a 1×1 submatrix, got k={best['k']}"
    assert best["R"] is not None and best["C"] is not None


if __name__ == "__main__":
    #import traceback

    # tests = [
    #     test_identity_k1,
    #     test_dense_full_rank,
    #     test_all_ones_rank1,
    #     test_embedded_block,
    #     test_block_diagonal_finds_larger,
    #     test_repeated_rows_rank1,
    #     test_float_valued,
    #     test_large_sparse_smoke,
    # ]

    # passed = failed = 0
    # for fn in tests:
    #     try:
    #         fn()
    #         print(f"  PASS  {fn.__name__}")
    #         passed += 1
    #     except Exception:
    #         print(f"  FAIL  {fn.__name__}")
    #         traceback.print_exc()
    #         failed += 1

    # print(f"\n{passed}/{passed + failed} tests passed.")
    # if failed:
    #     raise SystemExit(1)
    # MPI setup (graceful fallback to CLI args or single-process)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        _mpi_available = True
    except (ImportError, RuntimeError):
        comm = None
        _mpi_available = False
        rank = None  # resolved after argparse
        size = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='/mnt/home/lsaul/Datasets/flywire/connectome.txt')
    parser.add_argument('--int', action='store_true', help='S_value is integer-valued')
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--n_seeds', type=int, default=100)
    parser.add_argument('--var_threshold', type=float, default=0.99, help='SVD tail cutoff threshold')

    parser.add_argument('--rank', type=int, default=None, help='rank of this task (0-based); overrides MPI')
    parser.add_argument('--size', type=int, default=None, help='total number of tasks; overrides MPI')

    args = parser.parse_args()

    if not _mpi_available:
        rank = args.rank if args.rank is not None else 0
        size = args.size if args.size is not None else 1

    if args.filename.endswith('.mat'):
        mat = sio.loadmat(args.filename)
        rows = mat['rows'].ravel().astype(np.int64)   # 1-based from MATLAB
        cols = mat['cols'].ravel().astype(np.int64)
        vals = mat['vals'].ravel()
        S_index = np.vstack([rows, cols]) - 1         # convert to 0-based
        S_value = vals.astype(np.int64) if args.int else vals.astype(np.float64)
        m = n = int(rows.max())
    else:
        data = np.loadtxt(args.filename).T
        S_index = data[0:2, :].astype(np.int64)
        S_value = data[2, :]
        if args.int:
            S_value = S_value.astype(np.int64)
        m = n = int(S_index[0].max())
        # normalize to start at zero-index
        S_index -= 1
    if rank == 0:
        print(f'm=n={m}, E={len(S_value)}, size={size}, mpi={_mpi_available}')
        print(f'effective_rank with var_threshold={args.var_threshold}')

    # split restarts across ranks
    restarts_per_rank = args.n_restarts // size
    extra = args.n_restarts - restarts_per_rank * size
    local_restarts = restarts_per_rank + (1 if rank < extra else 0)
    print(f"size={size}, local_restarts={local_restarts}")

    start = time.time()
    local_best = find_large_positive_fullrank_square_multistart(
        S_index, S_value, m=m, n=n,
        core_thresholds=(5,),
        n_restarts=local_restarts,
        rng_seed=rank*100+1,          # different seed per rank - TODO: modified to avoid stuck runs?
        verbose=(rank == 0),
        n_seeds=args.n_seeds,
        var_threshold=args.var_threshold,
    )
    end = time.time()

    if comm is not None:
        all_results = comm.gather(local_best, root=0)
    else:
        all_results = [local_best]

    if rank == 0:
        print(all_results)
        best = max(all_results, key=lambda x: x["k"])
        print(f"algorithm find k={best['k']} in {(end-start):.4f} seconds!")
        ## for numerical precision -- TODO - DOUBLE-CHECK
        if args.var_threshold == 1.0:
            csr = sp.coo_matrix((S_value, (S_index[0], S_index[1])), shape=(m, n)).tocsr()
            A = csr[best["R"], :][:, best["C"]].toarray()
            best["k_stable"] = np.linalg.matrix_rank(A)
            print(f"\nSubmatrix A[R, C]  ({A.shape[0]}×{A.shape[1]}):")
            print(f"all positive = {np.all(A > 0)}, minimum = {A.min()}, rank = {best['k_stable']}") 
        print(best)

    # if best["k"] > 1:
    #     import scipy.sparse as sp
    #     rows_np = S_index[0].numpy().astype(np.int64)
    #     cols_np = S_index[1].numpy().astype(np.int64)
    #     vals_np = S_value.numpy().astype(np.float64)
    #     csr = sp.coo_matrix((vals_np, (rows_np, cols_np)), shape=(m, n)).tocsr()
    #     A = csr[best["R"], :][:, best["C"]].toarray()
    #     print(f"\nSubmatrix A[R, C]  ({A.shape[0]}×{A.shape[1]}):")
    #     with np.printoptions(precision=3, suppress=False, linewidth=240, threshold=np.inf):
    #         print(A)
