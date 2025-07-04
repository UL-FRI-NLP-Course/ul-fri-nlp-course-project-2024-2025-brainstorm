#!/bin/bash
#SBATCH --job-name=test-gams    # Job name
#SBATCH --partition=gpu         # Partition (queue) name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --cpus-per-task=8       # CPU cores/threads per task
#SBATCH --gres=gpu:1            # Number of GPUs per node (we only need 1 for testing)
#SBATCH --gpu-bind=none  # Try this if default binding fails

#SBATCH --mem=64G               # Job memory request
#SBATCH --time=01:00:00         # Time limit hrs:min:sec

# Standard output and error log
#SBATCH --output=/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/log_test/test-gams-%J.out
#SBATCH --error=/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/log_test/test-gams-%J.err

# Create testing directory if it doesn't exist
mkdir -p "/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/testing"

echo "Starting testing job $SLURM_JOB_ID on $HOSTNAME"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="/d/hpc/home/aj3477/.cache/huggingface"
export HF_HOME="/d/hpc/home/aj3477/.cache/huggingface"

# Run the test script
singularity exec --nv --overlay /d/hpc/projects/onj_fri/brainstorm/container/overlay_pp.img /d/hpc/projects/onj_fri/brainstorm/container/container_pp.sif \
    python3 /d/hpc/home/aj3477/NLP/FineTune/report_generation.py

echo "Testing job $SLURM_JOB_ID finished"