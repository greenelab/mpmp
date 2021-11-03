#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env
echo "Environment loaded"

DATA_TYPES=(
  "expression"
  "me_27k"
  "me_450k"
)
data_type=${DATA_TYPES[${SLURM_ARRAY_TASK_ID}]}

for seed in 42 1; do
    for n_dim in 10 100 500 1000 5000; do

        # raw data
        # results_dir=./06_predict_survival/results/me_baseline_alphas/results_${n_dim}_raw/
        # cmd="python 06_predict_survival/run_survival_prediction.py "
        # cmd+="--results_dir $results_dir "
        # cmd+="--seed $seed "
        # cmd+="--subset_mad_genes $n_dim "
        # cmd+="--overlap_data_types expression me_27k me_450k "
        # cmd+="--training_data ${data_type} "
        # echo "Running: $cmd"
        # eval $cmd

        # top n_dim PCA features
        results_dir=./06_predict_survival/results/me_baseline_alphas/results_${n_dim}_pca/
        cmd="python 06_predict_survival/run_survival_prediction.py "
        cmd+="--n_dim $n_dim "
        cmd+="--results_dir $results_dir "
        cmd+="--seed $seed "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="--training_data ${data_type} "
        echo "Running: $cmd"
        eval $cmd

    done
done

echo "Job complete"
