## Subzero Matrix Completion (sumac)

Python implementation of subzero matrix completion, a rectified linear factorization of a sparse matrix $S \in \mathbb R^{m \times n}$ as $S = \text{ReLU}(A B^{\top})$ such that $A, B \in \mathbb R^{m \times d}, d \ll \min(m,n)$.

We offer two major routines:
- Stochastic Alternating Least Squares Algorithm (sumac-SALSA): update factors alternately using random subsets of the other factor and its associated subset of edges.
- Gradient descent (sumac-GD): update factors $A, B$ simultaneously via minibatch gradient descent

### Installation

Create and activate a Python virtual environment. SUMAC currently requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
```

For a CPU-only install, install the CPU PyTorch wheel first, then install SUMAC:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

For a CUDA install, install a PyTorch wheel matching your CUDA runtime, then install SUMAC with the CUDA extra. For example, for CUDA 12.8:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[cuda]"
```

For development, install the developer extra:

```bash
python -m pip install -e ".[dev]"
```

or, for CUDA development:

```bash
python -m pip install -e ".[cuda,dev]"
```

The developer extra includes `pytest` for tests and `build` for creating source and wheel distributions:

```bash
python -m build
```

The package source is under ```src/sumac```, with examples in ```examples```.

### Python API

The main entry point is `sumac.sumac_factorize`. Inputs are passed by keyword. The sparse matrix is represented in COO form by `S_index` with shape `(2, nnz)` and `S_value` with shape `(nnz,)`.

```python
from sumac import sumac_factorize

A, B, history = sumac_factorize(
    S_index=S_index,
    S_value=S_value,
    shape=(m, n),
    rank=16,
    method="SALSA",
    max_iterations=100,
    num_blocks=100,
    device="cuda",
)
```

The return values are the learned factors `A` and `B`, with shapes `(m, rank)` and `(n, rank)`, plus a metric history collected during training. If `device` is omitted, SUMAC infers it from `S_index` and `S_value`; if `device` is provided, the sparse input tensors are moved there before training.

Common options are `method="SALSA"` or `method="GD"`, `rank`, `max_iterations`, `num_blocks`, `dtype`, `seed`, `momentum`, `learning_rate`, `optimizer`, `eval_interval`, `verbose`, and `allow_tf32`. CUDA kernel autotuning can be controlled with `autotune="cache"`, `"force"`, `"disable"`, or `"fallback"`; cached tuning results are stored under `$XDG_CACHE_HOME/sumac` by default, or `~/.cache/sumac` if `XDG_CACHE_HOME` is unset. Use `autotune_cache_dir` to override this location and `autotune_verbose=True` to print tuning decisions.

### Application: Fly Connectome Data
The connectome data is a large sparse matrix, with $m=n=139255, |S|=19773733$. The data is available at ```/mnt/home/lsaul/Datasets/flywire/connectome.txt```

To run sumac-SALSA
```bash
python examples/sumac_connectome.py --filename /path/to/connectome.txt --mode SALSA
```
To run sumac-GD
```bash
python examples/sumac_connectome.py --filename /path/to/connectome.txt --mode GD
```

### Exploration: Lower bound of the rank (embedding dimension)

Given a sparse matrix, what's the minimum (effective) rank needed for subzero matrix completion? We include a simple greedy algorithm in ```tests/check_min_rank.py```. To compute the (effective) rank for our example datasets, use ```check_min_rank.sh```.
