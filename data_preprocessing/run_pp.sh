#!/bin/bash
#SBATCH --job-name=test-gams    # Job name
#SBATCH --partition=gpu         # Partition (queue) name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks (processes)
#SBATCH --cpus-per-task=16       # CPU cores/threads per task
#SBATCH --gres=gpu:1            # Number of GPUs per node (we only need 1 for testing)
#SBATCH --mem=128G               # Job memory request
#SBATCH --time=20:00:00         # Time limit hrs:min:sec

# Standard output and error log
#SBATCH --output=logs/pp-test-%J.out
#SBATCH --error=logs/pp-test-%J.err

# Create testing directory if it doesn't exist
mkdir -p logs

echo "Starting testing job $SLURM_JOB_ID on $HOSTNAME"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Run the test script
singularity exec --nv --overlay /d/hpc/projects/onj_fri/brainstorm/container/overlay_pp.img /d/hpc/projects/onj_fri/brainstorm/container/container_pp.sif \
    python3 /d/hpc/home/aj3477/NLP/DataPreprocess/primerjava_w_sentences_BERT_w_QA.py

echo "Testing job $SLURM_JOB_ID finished"