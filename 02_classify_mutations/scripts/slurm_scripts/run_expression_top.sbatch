#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-268%10
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env
echo "Environment loaded"

# run command for a given gene in the list
# read the list of genes from tsv file
merged_filename="data/top_genes.tsv"
read_genes_from_file() {
    # create global gene array
    declare -a -g genes

    # read tab-separated file, genes should be the first column
    while IFS=$'\t' read -r gene class; do
        genes+=("$gene")
    done < "$1"

    # remove header
    genes=("${genes[@]:1}")
}
read_genes_from_file $merged_filename

gene=${genes[${SLURM_ARRAY_TASK_ID}]}

results_dir=02_classify_mutations/results/expression_top
n_feats=8000

for seed in 42 1; do
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${gene} "
    cmd+="--overlap_data_types expression "
    cmd+="--results_dir $results_dir "
    cmd+="--seed $seed "
    cmd+="--training_data expression "
    cmd+="--verbose"
    echo "Running: $cmd"
    eval $cmd
done

echo "Job complete"

