from contextlib import redirect_stdout
import torch

from sumac.config.options import make_config_from_args
from sumac.example_support._example_shared import example_type, eval_only, make_paths, load_factorize_save

def run_example(etype: example_type):
    config = make_config_from_args()
    if config.input_filename is None:
         raise ValueError("To run this example, please include the input data file using the --filename argument.")
       
    torch.set_float32_matmul_precision('high' if config.allow_tf32 else 'highest')

    ## TODO: Consider forcing verbose to on, forcing seed for example

    if config.eval_only:
        eval_only(config, etype)
        exit()
    
    (save_path, log_fn) = make_paths(config, etype)
    if log_fn is not None:
        with open(log_fn, 'w') as f:
            with redirect_stdout(f):
                load_factorize_save(config, etype, save_path)
    else:
        load_factorize_save(config, etype, save_path)
