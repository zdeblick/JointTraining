#!/bin/bash
#SBATCH --job-name=Joint_Training     # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=20gb                     # Job memory request (per node)
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/JT_%A_%a.log  #/allen/programs/braintv/workgroups/cortexmodels/daniel.zdeblick/JointTraining/logs
#SBATCH --partition braintv         # Partition used for processing
#SBATCH --array 0-6479
#SBATCH --chdir=/allen/programs/braintv/workgroups/cortexmodels/daniel.zdeblick/JointTraining

/home/daniel.zdeblick/anaconda3/bin/python3 digits_sim_script.py
