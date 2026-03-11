import numpy as np
import scipy.sparse as sp
from tests.test_sumac import generate_low_rank_data
import time 

# Given a (sparse) matrix S with shape (m,n), greedily search for a large positive submatrix
# that has the largest rank (or effective rank)
# Algorithm 1: find_large_positive_fullrank_square()
# Algorithm 2: greedy_positive_rectangle()

def build_pos_csr_csc(S_index: np.ndarray, S_value: np.ndarray, m: int, n: int):
    assert S_index.shape[0] == 2
    mask = S_value > 0
    ij = S_index[:, mask].astype(np.int64, copy=False)
    is_float = np.issubdtype(S_value.dtype, np.floating)
    #print(f"S is float = {is_float}")
    if is_float:
        v = S_value[mask].astype(np.float64, copy=False)
    else:
        v = S_value[mask].astype(np.int64, copy=False)
    coo = sp.coo_matrix((v, (ij[0], ij[1])), shape=(m, n))
    csr = coo.tocsr(); csr.sort_indices()
    csc = coo.tocsc(); csc.sort_indices()
    return csr, csc, is_float

def bipartite_core_peel(csr: sp.csr_matrix, csc: sp.csc_matrix, a: int, b: int):
    m, n = csr.shape
    row_deg = np.diff(csr.indptr).astype(np.int64, copy=False)
    col_deg = np.diff(csc.indptr).astype(np.int64, copy=False)
    alive_r = np.ones(m, dtype=bool)
    alive_c = np.ones(n, dtype=bool)
    rq = list(np.flatnonzero(row_deg < a))
    cq = list(np.flatnonzero(col_deg < b))
    while rq or cq:
        while rq:
            i = rq.pop()
            if not alive_r[i] or row_deg[i] >= a:
                continue
            alive_r[i] = False
            start, end = csr.indptr[i], csr.indptr[i + 1]
            cols = csr.indices[start:end]
            live = cols[alive_c[cols]]
            col_deg[live] -= 1
            cq.extend(live[col_deg[live] == b - 1].tolist())
        while cq:
            j = cq.pop()
            if not alive_c[j] or col_deg[j] >= b:
                continue
            alive_c[j] = False
            start, end = csc.indptr[j], csc.indptr[j + 1]
            rows = csc.indices[start:end]
            live = rows[alive_r[rows]]
            row_deg[live] -= 1
            rq.extend(live[row_deg[live] == a - 1].tolist())
    r_idx = np.flatnonzero(alive_r)
    c_idx = np.flatnonzero(alive_c)
    csr_sub = csr[r_idx][:, c_idx].tocsr(); csr_sub.sort_indices()
    csc_sub = csr_sub.tocsc(); csc_sub.sort_indices()
    return csr_sub, csc_sub, r_idx, c_idx

def greedy_positive_rectangle(
    csr: sp.csr_matrix,
    csc: sp.csc_matrix,
    seed_row: int,
    rng: np.random.Generator,
    s_cols: int = 64,
    per_col_cap: int = 2000,
    shortlist_L: int = 3000,
    max_steps: int = 10_000,
):
    m, n = csr.shape
    start, end = csr.indptr[seed_row], csr.indptr[seed_row + 1]
    C_list = csr.indices[start:end].copy()
    if C_list.size == 0:
        return [seed_row], np.array([], dtype=np.int32)
    C_mask = np.zeros(n, dtype=bool)
    C_mask[C_list] = True
    R = [int(seed_row)]
    R_mark = np.zeros(m, dtype=bool)
    R_mark[seed_row] = True

    for _ in range(max_steps):
        k_next = len(R) + 1
        if C_list.size < k_next:
            break
        s = min(s_cols, C_list.size)
        sampled_cols = C_list if s == C_list.size else rng.choice(C_list, size=s, replace=False)

        cand_rows_parts = []
        for j in sampled_cols:
            c0, c1 = csc.indptr[j], csc.indptr[j + 1]
            neigh = csc.indices[c0:c1]
            if neigh.size == 0:
                continue
            if neigh.size > per_col_cap:
                idx = rng.choice(neigh.size, size=per_col_cap, replace=False)
                neigh = neigh[idx]
            cand_rows_parts.append(neigh)
        if not cand_rows_parts:
            break

        cand_rows = np.concatenate(cand_rows_parts)
        cand_rows = cand_rows[~R_mark[cand_rows]]
        if cand_rows.size == 0:
            break

        uniq, counts = np.unique(cand_rows, return_counts=True)
        if uniq.size > shortlist_L:
            top_idx = np.argpartition(counts, -shortlist_L)[-shortlist_L:]
            uniq = uniq[top_idx]

        best_i = None
        best_overlap = -1
        for i in uniq:
            r0, r1 = csr.indptr[i], csr.indptr[i + 1]
            cols_i = csr.indices[r0:r1]
            if cols_i.size < k_next:
                continue
            overlap = int(C_mask[cols_i].sum())
            if overlap < k_next:
                continue
            if best_i is None or overlap > best_overlap:
                best_overlap = overlap
                best_i = int(i)
        if best_i is None:
            break

        r0, r1 = csr.indptr[best_i], csr.indptr[best_i + 1]
        cols_best = csr.indices[r0:r1]
        newC = cols_best[C_mask[cols_best]]
        if newC.size == 0:
            break

        R.append(best_i)
        R_mark[best_i] = True
        C_mask[:] = False
        C_mask[newC] = True
        C_list = newC

    return R, C_list

def extract_dense_block(csr: sp.csr_matrix, R: np.ndarray, C: np.ndarray, dtype=np.int64):
    k = len(R)
    assert len(C) == k
    A = np.empty((k, k), dtype=dtype)
    for ii, r in enumerate(R):
        r0, r1 = csr.indptr[r], csr.indptr[r + 1]
        idx = csr.indices[r0:r1]
        dat = csr.data[r0:r1]
        pos = np.searchsorted(idx, C)
        if np.any(pos >= idx.size) or np.any(idx[pos] != C):
            raise ValueError("Chosen (R,C) is not a complete positive rectangle (missing edges).")
        A[ii, :] = dat[pos]
    return A

def extract_dense_rect(csr: sp.csr_matrix, R: np.ndarray, C: np.ndarray, dtype=np.int64):
    """Like extract_dense_block but for a (possibly non-square) |R|×|C| rectangle."""
    A = np.empty((len(R), len(C)), dtype=dtype)
    for ii, r in enumerate(R):
        r0, r1 = csr.indptr[r], csr.indptr[r + 1]
        idx = csr.indices[r0:r1]
        dat = csr.data[r0:r1]
        pos = np.searchsorted(idx, C)
        if np.any(pos >= idx.size) or np.any(idx[pos] != C):
            raise ValueError("Chosen (R,C) is not a complete positive rectangle (missing edges).")
        A[ii, :] = dat[pos]
    return A

def rank_mod_p(A: np.ndarray, p: int):
    # Forward (row-echelon) Gaussian elimination over GF(p). Sufficient for rank
    # computation; does NOT produce reduced row echelon form.
    M = (A % p).astype(np.int64, copy=True)
    n = M.shape[0]
    rank = 0
    col = 0
    for row in range(n):
        while col < n:
            piv_rel = np.argmax(M[row:, col] % p != 0)
            piv = row + piv_rel
            if M[piv, col] % p == 0:
                col += 1
                continue
            if piv != row:
                M[[row, piv], :] = M[[piv, row], :]
            inv = pow(int(M[row, col] % p), -1, p)
            M[row, :] = (M[row, :] * inv) % p
            if row + 1 < n:
                factors = M[row + 1 :, col] % p
                if np.any(factors):
                    M[row + 1 :, :] = (M[row + 1 :, :] - factors[:, None] * M[row, :]) % p
            rank += 1
            col += 1
            break
        if col >= n:
            break
    return rank

def rank_and_pivots_mod_p(A: np.ndarray, p: int):
    """Forward Gaussian elimination over GF(p) on a (possibly non-square) m×n matrix.
    Returns (rank, pivot_row_indices, pivot_col_indices) as indices into A."""
    m, n = A.shape
    M = (A % p).astype(np.int64, copy=True)
    row_perm = np.arange(m)   # tracks which original row is at each position after swaps
    pivot_rows, pivot_cols = [], []
    cur_row = 0
    for col in range(n):
        if cur_row >= m:
            break
        piv_rel = np.argmax(M[cur_row:, col] % p != 0)
        piv = cur_row + piv_rel
        if M[piv, col] % p == 0:
            continue   # entire remaining column is zero mod p — skip
        if piv != cur_row:
            M[[cur_row, piv], :] = M[[piv, cur_row], :]
            row_perm[[cur_row, piv]] = row_perm[[piv, cur_row]]
        inv = pow(int(M[cur_row, col] % p), -1, p)
        M[cur_row, :] = (M[cur_row, :] * inv) % p
        if cur_row + 1 < m:
            factors = M[cur_row + 1:, col] % p
            if np.any(factors):
                M[cur_row + 1:, :] = (M[cur_row + 1:, :] - factors[:, None] * M[cur_row, :]) % p
        pivot_rows.append(int(row_perm[cur_row]))
        pivot_cols.append(col)
        cur_row += 1
    r = len(pivot_rows)
    return r, np.array(pivot_rows, dtype=np.int64), np.array(pivot_cols, dtype=np.int64)

def certify_full_rank_integer(A: np.ndarray, primes=(1_000_000_007, 1_000_000_009, 998_244_353)):
    # If rank_mod_p(A, p) == n for any prime p, then det(A) is not divisible by p,
    # hence det(A) != 0 over Z, hence A has full integer rank. One confirming prime
    # is sufficient; multiple primes guard against the rare case where det(A) happens
    # to be divisible by the first prime tried.
    n = A.shape[0]
    for p in primes:
        if rank_mod_p(A, p) == n:
            return True, p
    return False, None

def certify_full_rank_float(A: np.ndarray, tol: float = None):
    """Certify full rank for a float matrix via SVD. `tol` is passed to np.linalg.matrix_rank."""
    return np.linalg.matrix_rank(A, tol=tol) == A.shape[0], "svd"

def rank_and_pivots_float(A: np.ndarray, tol: float = None):
    """Gaussian elimination with partial pivoting on a (possibly non-square) float matrix.
    Returns (rank, pivot_row_indices, pivot_col_indices) as indices into A."""
    m, n = A.shape
    M = A.astype(np.float64, copy=True)
    norm = np.abs(M).max()
    if tol is None:
        tol = max(m, n) * np.finfo(np.float64).eps * norm if norm > 0 else 1e-12
    row_perm = np.arange(m)
    pivot_rows, pivot_cols = [], []
    cur_row = 0
    for col in range(n):
        if cur_row >= m:
            break
        piv_rel = np.argmax(np.abs(M[cur_row:, col]))
        piv = cur_row + piv_rel
        if abs(M[piv, col]) < tol:
            continue   # column is numerically zero — skip
        if piv != cur_row:
            M[[cur_row, piv], :] = M[[piv, cur_row], :]
            row_perm[[cur_row, piv]] = row_perm[[piv, cur_row]]
        M[cur_row, :] /= M[cur_row, col]
        if cur_row + 1 < m:
            factors = M[cur_row + 1:, col].copy()
            M[cur_row + 1:, :] -= factors[:, None] * M[cur_row, :]
        pivot_rows.append(int(row_perm[cur_row]))
        pivot_cols.append(col)
        cur_row += 1
    r = len(pivot_rows)
    return r, np.array(pivot_rows, dtype=np.int64), np.array(pivot_cols, dtype=np.int64)


def rank_and_pivots_svd_effective(A: np.ndarray, variance_threshold: float = 0.99):
    """SVD-based effective rank + RRQR pivot selection (float or integer input).
    Effective rank = fewest singular values explaining >= variance_threshold of total variance.
    Returns (r, pivot_row_indices, pivot_col_indices)."""
    from scipy.linalg import qr as scipy_qr
    M = A.astype(np.float64, copy=False)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)  # s: length min(m, n)
    if s[0] == 0:
        return 0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    r = int(np.searchsorted(cumvar, variance_threshold) + 1)
    r = min(r, M.shape[0], M.shape[1])
    # DEIM-style pivot selection via RRQR on the dominant singular subspaces
    _, _, col_perm = scipy_qr(Vt[:r, :], pivoting=True)   # r x n  -> pivot columns
    _, _, row_perm = scipy_qr(U[:, :r].T, pivoting=True)   # r x m  -> pivot rows
    return (r,
            np.array(row_perm[:r], dtype=np.int64),
            np.array(col_perm[:r], dtype=np.int64))


def find_large_positive_fullrank_square(
    S_index: np.ndarray,
    S_value: np.ndarray,
    m: int,
    n: int,
    *,
    core_thresholds=(2,),   # for small test
    n_seeds=100,
    seed_mix_top=0.5,
    s_cols=64,
    per_col_cap=2000,
    shortlist_L=3000,
    max_steps=10_000,
    rng_seed=0,
    primes=(1_000_000_007, 1_000_000_009, 998_244_353),
    pivot_extraction=False,
    var_threshold=float,
):
    rng = np.random.default_rng(rng_seed)
    csr, csc, is_float = build_pos_csr_csc(S_index, S_value, m, n)
    block_dtype = np.float64 if is_float else np.int64
    # Trivial k=1 guarantee: any single positive entry is a full-rank 1×1 submatrix.
    # Without this, the algorithm returns k=0 when every greedy rectangle is large
    # but rank-deficient (e.g. all-ones or all-identical-rows matrices).
    if csr.nnz > 0:
        rows_init, cols_init = csr.nonzero()
        best = {"k": 1, "R": np.array([rows_init[0]]), "C": np.array([cols_init[0]]),
                "prime": "trivial", "core": "trivial"}
    else:
        best = {"k": 0, "R": None, "C": None, "prime": None, "core": None}
    # Process cores first (densest first) so best["k"] rises quickly, enabling more pruning
    # in later passes over sparser graphs.
    candidates = []
    for a in sorted(core_thresholds, reverse=True):
        csr2, csc2, r_map, c_map = bipartite_core_peel(csr, csc, a=a, b=a)
        if csr2.nnz > 0 and csr2.shape[0] > 0 and csr2.shape[1] > 0:
            candidates.append((f"core({a},{a})", csr2, csc2, r_map, c_map))
    candidates.append(("full", csr, csc, np.arange(m), np.arange(n)))

    max_possible_k = min(m, n)
    for tag, csr_g, csc_g, r_map, c_map in candidates:
        if best["k"] >= max_possible_k:
            break
        mm, nn = csr_g.shape
        if mm == 0 or nn == 0 or csr_g.nnz == 0:
            continue
        deg = np.diff(csr_g.indptr).astype(np.int64, copy=False)
        if deg.max() == 0:
            continue
        topK = min(int(n_seeds * seed_mix_top), mm)
        randK = n_seeds - topK
        top_rows = np.argsort(deg)[-topK:] if topK > 0 else np.array([], dtype=np.int64)
        rand_rows = rng.integers(0, mm, size=randK, endpoint=False) if randK > 0 else np.array([], dtype=np.int64)
        seeds = np.unique(np.concatenate([top_rows, rand_rows]).astype(np.int64, copy=False))
        # Sort seeds by descending degree so the `deg <= best["k"]` pruning guard
        # kicks in as early as possible.
        seeds = seeds[np.argsort(deg[seeds])[::-1]]

        for seed_row_local in seeds:
            if deg[seed_row_local] <= best["k"]:
                continue
            R_local, C_local = greedy_positive_rectangle(
                csr_g, csc_g, int(seed_row_local), rng,
                s_cols=s_cols, per_col_cap=per_col_cap,
                shortlist_L=shortlist_L, max_steps=max_steps
            )
            if pivot_extraction:
                # --- Pivot-extraction path [Default] ---
                # Operate on the full |R|×|C| greedy rectangle. SVD effective rank +
                # RRQR pivot selection identifies exactly which rows and columns form
                # the largest certified full-rank submatrix.
                if min(len(R_local), int(C_local.size)) <= best["k"]:
                    continue
                R_local_arr = np.array(R_local, dtype=np.int64)
                C_local_arr = np.array(C_local, dtype=np.int64)
                try:
                    A_rect = extract_dense_rect(csr_g, R_local_arr, C_local_arr, dtype=block_dtype)
                except ValueError:
                    continue
                r, pr, pc = rank_and_pivots_svd_effective(A_rect, variance_threshold=var_threshold)
                p_used = "svd_effective"
                if r > best["k"] and pr is not None:
                    best = {"k": r,
                            "R": r_map[R_local_arr[pr]],
                            "C": c_map[C_local_arr[pc]],
                            "prime": p_used, "core": tag}
            else:
                # --- Original path [OLD, not used] ---
                k0 = min(len(R_local), int(C_local.size))
                if k0 <= best["k"] or k0 == 0:
                    continue
                Rk_local = np.array(R_local[:k0], dtype=np.int64)
                Ck_local = np.array(C_local[:k0], dtype=np.int64)
                try:
                    A = extract_dense_block(csr_g, Rk_local, Ck_local, dtype=block_dtype)
                except ValueError:
                    continue
                if is_float:
                    ok, p = certify_full_rank_float(A)
                else:
                    ok, p = certify_full_rank_integer(A, primes=primes)
                if ok:
                    best = {"k": k0, "R": r_map[Rk_local], "C": c_map[Ck_local], "prime": p, "core": tag}
    return best


def find_large_positive_fullrank_square_multistart(
    S_index: np.ndarray,
    S_value: np.ndarray,
    m: int,
    n: int,
    *,
    n_restarts: int = 10,
    rng_seed: int = 0,
    verbose: bool = False,
    var_threshold: bool = True,
    **kwargs,
):
    """Run find_large_positive_fullrank_square with n_restarts different random seeds
    and return the result with the largest k found across all runs."""
    parent_rng = np.random.default_rng(rng_seed)
    seeds = parent_rng.integers(0, 2**31, size=n_restarts)
    best = {"k": 0, "R": None, "C": None, "prime": None, "core": None}
    for i, seed in enumerate(seeds):
        result = find_large_positive_fullrank_square(
            S_index, S_value, m, n, rng_seed=int(seed), 
            pivot_extraction=True, #TODO: check
            var_threshold=var_threshold,
            **kwargs
        )
        if verbose:
            print(f"  restart {i+1}/{n_restarts} (seed={seed}): k={result['k']}")
        if result["k"] > best["k"]:
            best = result
    return best


# --- Construct a small test matrix with a guaranteed positive, full-rank 3x3 submatrix ---

# m = n = 6
# S_dense = np.array([
#     [ 6,  2,  3,  0,  0,  0],
#     [ 1,  4,  5,  0,  0,  0],
#     [ 7,  8, 10,  0,  0,  0],
#     [ 0,  0,  0,  9,  0,  0],
#     [ 0,  0,  0,  0, 11,  0],
#     [ 0,  0,  0,  0,  0, 13],
# ], dtype=np.int64)

# The top-left 3x3 block is strictly positive and full rank:
# det3 = int(round(np.linalg.det(S_dense[:3,:3])))
# print(det3)

# Build COO (S_index, S_value) from dense
# rows, cols = np.nonzero(S_dense)
# vals = S_dense[rows, cols]
# S_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
# S_value = torch.tensor(vals, dtype=torch.long)

# Run pipeline
# best = find_large_positive_fullrank_square(
#     S_index, S_value, m=m, n=n,
#     core_thresholds=(2,),  # peel to nodes with degree>=2 on each side
#     n_seeds=200,
#     rng_seed=123,
# )

# m = n = 1000
# d_true = 16
# target_density = 0.1
# S_index, S_value, m, n = generate_low_rank_data(m, n, d_true, density=target_density, noise_level=0.0)

# best = find_large_positive_fullrank_square(
#     S_index, S_value, m=m, n=n,
#     core_thresholds=(5,),  # peel to nodes with degree>=2 on each side
#     n_seeds=200,
#     rng_seed=123,
# )

# print(best)

