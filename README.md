## Subzero Matrix Completion (sumac)

Python implementation of subzero matrix completion, a rectified linear factorization of a sparse matrix $S \in \mathbb R^{m \times n}$ as $S = \text{ReLU}(A B^{\top})$ such that $A, B \in \mathbb R^{m \times d}, d \ll \min(m,n)$.

We offer two major routines:
- Stochastic Alternating Least Squares Algorithm (sumac-SALSA): update factors alternately using random subsets of the other factor and its associated subset of edges.
- Gradient descent (sumac-GD): update factors $A, B$ simultaneously via minibatch gradient descent

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
The package source is under ```src/sumac```, with examples in ```examples```.

### Application: Fly Connectome Data
The connectome data is a large sparse matrix, with $m=n=139255, |S|=19773733$. The data is available at ```/mnt/home/lsaul/Datasets/flywire/connectome.txt```

To run sumac-SALSA
```
python sumac_connectome.py --mode SALSA
```
To run sumac-GD
```
python sumac_connectome.py --mode GD
```

### Exploration: Lower bound of the rank (embedding dimension)

Given a sparse matrix, what's the minimum (effective) rank needed for subzero matrix completion? We include a simple greedy algorithm in ```tests/check_min_rank.py```. To compute the (effective) rank for our example datasets, use ```check_min_rank.sh```.
