## Subzero Matrix Completion (sumac)

Python implementation of subzero matrix completion, a rectified linear factorization of a sparse matrix $S \in \mathbb R^{m \times n}$ as
$$
S = \text{ReLU}(A B^{\top}), \quad A, B \in \mathbb R^{m \times d}, d \ll \min(m,n) 
$$

We offer two major routines:
- Alternating Least Sqaures (sumac-ALS): update factors $A,B$ alternatively by solving least squares, together with an auxiliary variable $Z$ where $\min_{Z, A, B} \| Z - A B^{\top} \|_F^2, S = \text{ReLU}(Z)$
- Gradient descent (sumac-GD): update factors $A, B$ simultaneously via minibatch gradient descent (loss aligned with sumac-ALS)

### Setup and Overview
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
To run sumac-GD with loss aligned as ALS
```
python sumac_connectome.py --mode GDlatent_sumac
```
To run sumac-GD with loss aligned as ALS and preconditioning
```
python sumac_connectome.py --mode GDlatent_prec_sumac
```