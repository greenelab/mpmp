#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env
echo "Environment loaded"

RESULTS_DIR=./05_classify_mutations_multimodal/results/raw_shuffle_cancer_type
GENES=(
  "TP53"
  "KRAS"
  "EGFR"
  "PIK3CA"
  "IDH1"
  "SETD2"
)
NUM_FEATS=20000
gene=${GENES[${SLURM_ARRAY_TASK_ID}]}

for seed in 42 1; do

    # run all combinations of expression, me_27k, me_450k
    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k "
    cmd+="--subset_mad_genes $NUM_FEATS "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_450k "
    cmd+="--subset_mad_genes $NUM_FEATS "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k me_450k "
    cmd+="--subset_mad_genes $NUM_FEATS "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k me_450k "
    cmd+="--subset_mad_genes $NUM_FEATS "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    echo "Running: $cmd"
    eval $cmd

done

echo "Job complete"
