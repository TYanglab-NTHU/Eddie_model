#!/bin/bash

#SBATCH --job-name=Eddie
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=3
#SBATCH --account=MST111483
#SBATCH --mail-type=END,FAIL       
#SBATCH --mail-user=qteddie@gmail.com
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt    



python model_train.py