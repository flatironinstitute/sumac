from contextlib import redirect_stdout
import torch 

from sumac.data import *
from sumac.config.options import make_config_from_args

from ._example_shared import eval_only, make_paths, load_factorize_save

if __name__ == '__main__':
    config = make_config_from_args()
    if config.input_filename is None:
        raise ValueError("To run this example, please include the input data file using the --filename argument.")

    torch.set_float32_matmul_precision('high' if config.allow_tf32 else 'highest')
    
    if config.eval_only:
        eval_only(config, 'digits')
        exit()

    (save_path, log_fn)  = make_paths(config, "digits")
    if log_fn is not None:
        with open (log_fn, 'w') as f:
            with redirect_stdout(f):
                load_factorize_save(config, 'digits', save_path)
    else:
        load_factorize_save(config, 'digits', save_path)
    
##launch scripts
#GPU: python sumac_digits.py --iters 1000  --num_blocks 100 
#EVAL: python sumac_digits.py --eval_only
