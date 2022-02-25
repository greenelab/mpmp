#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env2
echo "Environment loaded"

# run command for a given gene in the list
# top 15 genes by predictive performance using gene expression
genes=(
  "FUBP1"
  "TSC1"
  "PTEN"
  "BRAF"
  "NFE2L2"
  "NRAS"
  "SPOP"
  "FBXW7"
  "PIK3CA"
  "CTNNB1"
  "CDKN2A"
  "ERBB2"
  "EGFR"
  "TP53"
  "RB1"
)

gene=${genes[${SLURM_ARRAY_TASK_ID}]}
num_feats=8000

for seed in 42 1; do
    results_dir=./02_classify_mutations/results/dosage_effects/only_target
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes $gene "
    cmd+="--overlap_data_types expression "
    cmd+="--results_dir $results_dir "
    cmd+="--subset_mad_genes $num_feats "
    cmd+="--seed $seed "
    cmd+="--training_data expression "
    cmd+="--only_target "
    echo "Running: $cmd"
    eval $cmd
done

echo "Job complete"
