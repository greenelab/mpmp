#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-24
#SBATCH --cpus-per-task=3
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env2
echo "Environment loaded"

RESULTS_DIR=./05_classify_mutations_multimodal/results/bayes_opt
GENES=(
  "TP53"
  "KRAS"
  "EGFR"
  "PIK3CA"
  "IDH1"
  "SETD2"
)

FOLDS=(
  '0'
  '1'
  '2'
  '3'
)

fold_ix=$((${SLURM_ARRAY_TASK_ID} % 4))
gene_ix=$((${SLURM_ARRAY_TASK_ID} / 4))
fold=${FOLDS[$fold_ix]}
gene=${GENES[$gene_ix]}

for seed in 42 1; do

    # run all combinations of expression, me_27k, me_450k
    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--bayes_opt "
    cmd+="--bayes_opt_fold_no $fold "
    cmd+="--save_hparams "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--bayes_opt "
    cmd+="--bayes_opt_fold_no $fold "
    cmd+="--save_hparams "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--bayes_opt "
    cmd+="--bayes_opt_fold_no $fold "
    cmd+="--save_hparams "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--bayes_opt "
    cmd+="--bayes_opt_fold_no $fold "
    cmd+="--save_hparams "
    echo "Running: $cmd"
    eval $cmd

done

echo "Job complete"
