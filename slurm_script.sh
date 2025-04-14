#!/bin/bash
#SBATCH --job-name=Semantic_matcher_train
#SBATCH --nodes=1 # Request 1 nodes
#SBATCH --ntasks=1 # Nº of Processes per node
#SBATCH --cpus-per-task=2 # Requests 2 CPUs to assist in each task
#SBATCH --partition=gpuPartition # Partition
#SBATCH --output=%x_%j.out      ### Slurm Output file, %x is job name, %j is job id
#SBATCH --mem=8 # memory reserved for the job

# Load the Python environment
source venv/bin/activate

# List of scripts/values to run

# Run the script/values corresponding to the array index
srun python3 train.py

#$SLURM_ARRAY_TASK_ID defines which array position is running

#If there is only one script/value combination to run, remove the (#SBATCH --array=0-5)
#And run something like
#srun python3 code/run.py --hyper hyperparameters/Tabular/Abalone/