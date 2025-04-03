#!/bin/bash
#SBATCH -p gpu-long            # Partition (queue)
#SBATCH -A kdss                # Account
#SBATCH --cpus-per-task=64     # Number of CPUs
#SBATCH --mem=64GB             # Memory
#SBATCH --time=100:00:00       # Time limit (2 hours)
#SBATCH --gres=gpu:L40:1       # Request 1 L40 GPU
#SBATCH -o %j.slurm.out              # Standard output

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"

PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
echo "Timestamp: $TIMESTAMP"

export PROCESS_COUNT=$SLURM_CPUS_ON_NODE
echo "Using $PROCESS_COUNT parallel processes"

cd $PROJECT_DIR
python run-multiple.py

echo "Job completed at $(date)"
