#!/bin/bash -l

DATASET_DIR=$SCRATCH"/Files/GitRepos/Effective-Connectivity-Reservoir-Computing/Datasets/stroke/"
TOOLS_DIR=$SCRATCH"/Files/GitRepos/masters-thesis/tools/"

cd DATASET_DIR

TIMESTAMP=$(date +%y-%m-%d_%H%M)

# The file with the list of subjects to analize
subjects_list="subjects_list.txt"

# Read the subjects names from the subjects list
while read subject; do
  # Process the subject
  sbatch $TOOLS_DIR"slurm/rcc-stroke/launch.sh" $subject $TIMESTAMP
done < "$subjects_list"
