#!/bin/bash
#SBATCH --mail-user=thuang@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH -J run_connectome_v2
#SBATCH -o connectome/sbatch_logs/v2.o%j
#SBATCH -e connectome/sbatch_logs/v2.e%j
# Optional: load modules or activate your Python environment
module --force purge
source /mnt/home/thuang/sumac/.venv/bin/activate
#module spider cuda

# echo out for logs
echo "Running on node $(hostname)"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "GPUs per task: $SLURM_GPUS_PER_TASK"
echo "Before run:" && free -h
#python sumac_connectome.py --iters 1001 --d 16 --num_blocks 50 --mode SALSA --momentum 0.9
#python sumac_connectome.py --iters 1001 --d 16 --num_blocks 100 --mode SALSA --momentum 0.9
#CUDA_LAUNCH_BLOCKING=1 python sumac_connectome.py --iters 55 --d 16
#python sumac_connectome.py --iters 54 --d 16

#module spider cuda/12.5.1
module load modules/2.4-20250724
module load cuda/12.5.1
nsys profile --trace=cuda,nvtx --cuda-memory-usage true --python-sampling true -f true -o nsys_logs/sumac_profile_gd_new_3gpu_0223 python sumac_connectome.py --iters 55
echo "After run:" && free -h