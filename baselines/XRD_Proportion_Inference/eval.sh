#!/bin/bash
#SBATCH --job-name=xrd_eval
#SBATCH --partition=your-partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:1             # Use one GPU

# Load the environment
# module load miniconda
# source activate xrd

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run evaluation
# Reads best_proportion_model.pth by default
echo "Starting evaluation at $(date)"

for num_phases in 2 3 4
do
    echo "Running evaluation for $num_phases phases..."
    python -u Neural_network/test.py --num_phases $num_phases
done

echo "Evaluation finished at $(date)"
