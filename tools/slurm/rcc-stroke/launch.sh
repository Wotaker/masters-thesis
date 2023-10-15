#!/bin/bash -l

#SBATCH --job-name="rcc-stroke"
#SBATCH -A plgsano4-cpu
#SBATCH --partition plgrid

#SBATCH --output="/net/ascratch/people/plgwciezobka/Logs/rcc-stroke_27-10-23/%j-%x-output.out"
#SBATCH --error="/net/ascratch/people/plgwciezobka/Logs/rcc-stroke_27-10-23/%j-%x-error.err"

#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=1GB

SUBJECT=$1
TIMESTAMP=$2
GITREPO="Effective-Connectivity-Reservoir-Computing"
FOLDER_SPEC="rcc-stroke"

# === RCC method parameters ===

jobs="1"
length="100"
split="90"
skip="10"
runs="20"
surrogates="100"
min_lag="-3"
max_lag="0"
rois="-1"
plots="false"

# === Execution ====

echo "[Debug] Subject: $SUBJECT"
data_dir="Datasets/stroke"
results_dir="Results_"$FOLDER_SPEC"_"$TIMESTAMP

conda activate $SCRATCH/venvs/rcc-conda-310
echo "[Debug] Env activated"
cd $SCRATCH/Files/GitRepos/Effective-Connectivity-Reservoir-Computing/
python -u main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $SUBJECT --rois $rois --num_surrogates $surrogates --runs $runs --min_lag $min_lag --max_lag $max_lag fmri --plots $plots
echo "[Debug] Deactivating env"
conda deactivate
