#!/bin/bash
#SBATCH --job-name=finetune-gams    # Job name
#SBATCH --partition=gpu              # Partition (queue) name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=4            # CPU cores/threads per task (increased for fine-tuning)
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --mem=32G                    # Job memory request (increased for fine-tuning) - added G for GB
#SBATCH --time=2:00:00              # Time limit hrs:min:sec (increased for training)

# Standard output and error log
#SBATCH --output=/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/log_finetune/finetune-gams-%J.out
#SBATCH --error=/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/log_finetune/finetune-gams-%J.err

# Create model output directories
mkdir -p "/d/hpc/projects/onj_fri/brainstorm/FineTune_models/GaMS-9B-Instruct/checkpoints"
mkdir -p "/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct"
mkdir -p "/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct/testing"

echo "Starting fine-tuning job $SLURM_JOB_ID on $HOSTNAME"
echo "Requesting $SLURM_CPUS_PER_TASK CPUs and $SLURM_MEM_PER_NODE MB RAM"
echo "Requesting $SLURM_GPUS_ON_NODE GPU(s)"

# Set environment variables to optimize GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0        # Changed to use only GPU 0
export TRANSFORMERS_CACHE="/d/hpc/home/aj3477/.cache/huggingface"
export HF_HOME="/d/hpc/home/aj3477/.cache/huggingface"

# Run the fine-tuning script in the container
singularity exec --nv --overlay /d/hpc/projects/onj_fri/brainstorm/container/overlay.img /d/hpc/projects/onj_fri/brainstorm/container/container_llm.sif \
    python3 /d/hpc/home/aj3477/NLP/FineTune/finetune_GaMS_9B.py

echo "Fine-tuning job $SLURM_JOB_ID finished"