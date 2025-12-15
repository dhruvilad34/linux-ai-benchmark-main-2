#!/bin/bash
#SBATCH --job-name=ai_benchmark
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=32G

# Load necessary modules (adjust based on your HPC setup)
# module load cuda/12.2
# module load python/3.10

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source venv/bin/activate

# Run benchmark
echo "Starting benchmark at $(date)"
python main.py --config config/config.yaml --num_tasks 100

echo "Benchmark completed at $(date)"


