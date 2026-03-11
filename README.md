## Subzero Matrix Completion (sumac)

Python implementation of subzero matrix completion, a rectified linear factorization of a sparse matrix $S \in \mathbb R^{m \times n}$ as $S = \text{ReLU}(A B^{\top})$ such that $A, B \in \mathbb R^{m \times d}, d \ll \min(m,n)$.

We offer three major routines (all of which support CPU/Single GPU/Multi-GPU):
- Gradient descent (sumac-GD): update factors $A, B$ simultaneously via minibatch gradient descent (loss aligned with sumac-ALS)
- Alternating Least Squares (sumac-ALS): update factors $A,B$ alternatively by solving least squares, together with an auxiliary variable $Z$ where $\min_{Z, A, B} || Z - A B^{\top} ||_F^2, S = \text{ReLU}(Z)$
- Stochastic Alternating Least Squares Algorithm (sumac-SALSA): similar to ALS to update factors alternatively, but only use a random subset of the other factor (and its associated subset of edges). Preferred to ALS for faster convergence in large $S$.

### Setup and Overview
Create a Python virtual environment (for Python > 3.10) and activate it
```
python -m venv .venv
source .venv/bin/activate
```
Then install the dependencies
```
pip install -r requirements.txt
```
The driver code is in ```sumac.py```, which routines and helper functions in ```_sumac```. 

### Application: Fly Connectome Data
The connectome data is a large sparse matrix, with $m=n=139255, |S|=19773733$. The data is available at ```/mnt/home/lsaul/Datasets/flywire/connectome.txt```

To run sumac-ALS
```
python sumac_connectome.py --mode ALS
```
To run sumac-SALSA
```
python sumac_connectome.py --mode SALSA
```
To run sumac-GD with loss aligned as ALS/SALSA
```
python sumac_connectome.py --mode GDlatent_sumac
```
To run sumac-GD with loss aligned as ALS and preconditioning
```
python sumac_connectome.py --mode GDlatent_prec_sumac
```

### Exploration: Lower bound of the rank (embedding dimension)

Given a sparse matrix, what's the minimum (effective) rank needed for subzero matrix completion? We include a simple greedy algorithm in ```tests/check_min_rank.py```. To compute the (effective) rank for our example datasets, use ```check_min_rank.sh```.