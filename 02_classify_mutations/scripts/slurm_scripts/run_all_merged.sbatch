#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH --array=0-444%15
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env2
echo "Environment loaded"

# read the list of cosmic genes from tsv file
merged_filename="data/cancer_genes/merged_with_annotations.tsv"

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

results_dir=02_classify_mutations/results/merged_all
n_feats=8000
n_dim=5000

for seed in 42 1; do

    # use raw data for non-methylation data types
    for data_type in mirna mut_sigs rppa expression; do
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes ${gene} "
        cmd+="--results_dir ${results_dir} "
        cmd+="--training_data ${data_type} "
        cmd+="--seed ${seed} "
        cmd+="--num_features ${n_feats} "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        echo "Running: $cmd"
        eval $cmd
    done

    # use compressed data for methylation
    for data_type in me_27k me_450k; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes ${gene} "
        cmd+="--results_dir ${results_dir} "
        cmd+="--training_data ${data_type} "
        cmd+="--seed ${seed} "
        cmd+="--n_dim ${n_dim} "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        echo "Running: $cmd"
        eval $cmd
    done

done
echo "Job complete"

