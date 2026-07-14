from sumac.example_support.example_core import run_example

if __name__ == '__main__':
    run_example('bigrams')

##launch scripts
#GPU: python sumac_bigrams.py --iters 1000  --num_blocks 100 
#EVAL: python sumac_bigrams.py --eval_only
