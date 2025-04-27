#!/bin/bash
#SBATCH --job-name=news-gen    # Job name
#SBATCH --partition=gpu              # Partition (queue) name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # CPU cores/threads per task (Increased)
#SBATCH --gres=gpu:2                 # Number of GPUs per node (Ensure GPU has enough VRAM, e.g., >40GB, ideally 80GB)
#SBATCH --mem=128G                   # Job memory request (Increased significantly for 17B model)
#SBATCH --time=02:00:00              # Time limit hrs:min:sec (May need adjustment)

# Standard output and error log
#SBATCH --output=logs/news-gen-%J.out
#SBATCH --error=logs/news-gen-%J.err

# Create logs directory if it doesn't exist
mkdir -p logs
# Create directory for generated reports if it doesn't exist
mkdir -p /d/hpc/projects/onj_fri/brainstorm/generated_reports

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"
echo "Requesting $SLURM_CPUS_PER_TASK CPUs and $SLURM_MEM_PER_NODE MB RAM"
echo "Requesting $SLURM_GPUS_ON_NODE GPU(s)"

# Run the model test script in the container
# Ensure python3 points to the correct environment within the container
# Ensure news_generation.py is accessible (usually CWD is mounted)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
singularity exec --nv --overlay container/overlay.img container/container_llm.sif python3 news_generation.py

# echo "--- Testing basic python execution in singularity ---"

# # TEST COMMAND: Replace the original script execution with this:
# singularity exec --nv --overlay container/overlay2.img container/container_llm2.sif python3 -c 'import sys; print("--- Python via singularity exec works ---", sys.version, flush=True)'

# echo "--- Test finished ---"

echo "Job $SLURM_JOB_ID finished"
