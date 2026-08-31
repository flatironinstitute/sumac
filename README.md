## Subzero Matrix Completion (sumac)

Python implementation of subzero matrix completion, a rectified linear factorization of a sparse matrix $S \in ℝ^{m \times n}$ as $S \approx \text{ReLU}(A B^{\top})$ such that $A\in ℝ^{m \times d}, B\in ℝ^{n \times d}, d \ll \min(m,n)$.

We offer two major routines:
- Stochastic Alternating Least Squares Algorithm (sumac-SALSA): update factors alternately using random subsets of the other factor and its associated subset of edges.
- Gradient descent (sumac-GD): update factors $A, B$ simultaneously via minibatch gradient descent

### Installation

Create and activate a Python virtual environment. SUMAC currently requires Python 3.12 or newer.

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

For a ROCm install, first install a ROCm build of PyTorch together with the
matching Triton package supplied by that PyTorch distribution, then install SUMAC with the ROCm extra. For example, for ROCm 7.2:

```bash
python -m pip install "torch==2.13.0+rocm7.2" --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install -e ".[rocm]"
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

The main entry point is `sumac.sumac_factorize`. Its arguments are keyword-only. The sparse matrix is represented in COO form by `S_index` with shape `(2, nnz)` and `S_value` with shape `(nnz,)`. Factorization settings are collected in a `SumacConfig` object.

```python
import torch

from sumac import SumacConfig, SumacMethod, sumac_factorize

config = SumacConfig(
    method=SumacMethod.SALSA,
    rank=16,
    max_iterations=100,
    num_blocks=100,
    device=torch.device("cuda"),
)

A, B, history = sumac_factorize(
    S_index=S_index,
    S_value=S_value,
    shape=(m, n),
    config=config,
)
```

Optional `A_init` and `B_init` keyword arguments can supply initial factor tensors. The return values are the learned factors `A` and `B`, with shapes `(m, config.rank)` and `(n, config.rank)`, plus a metric history collected during training. If `config.device` is `None`, SUMAC infers the device from `S_index` and `S_value`; if it is set, the sparse input tensors are moved there before training.

`SumacConfig` is keyword-only and provides defaults for every field. See its docstring for the complete list of configuration options and their behavior.

### Application: Fly Connectome Data

The connectome data is a large sparse matrix, with $m=n=139255, |S|=19773733$. 

To run sumac-SALSA
```bash
python examples/sumac_connectome.py --filename /path/to/connectome_data --mode SALSA
```
To run sumac-GD
```bash
python examples/sumac_connectome.py --filename /path/to/connectome_data --mode GD
```


Data for the examples can be found at:
https://users.flatironinstitute.org/~lsaul/sparse_matrices/
