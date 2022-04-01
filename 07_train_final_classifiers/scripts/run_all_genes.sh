#!/bin/bash
PARAMS_DIR=./02_classify_mutations/results/merged_filter_all
RESULTS_DIR=./07_train_final_classifiers/results/merged_genes
ERRORS_DIR=./final_merged_errors
SEED=42

mkdir -p $ERRORS_DIR

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

for gene in "${genes[@]}"; do

    # just do expression for now
    # TODO: could do methylation but would need to pass PCA features
    cmd="python 07_train_final_classifiers/train_classifier.py "
    cmd+="--gene $gene "
    cmd+="--params_dir $PARAMS_DIR "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression "
    cmd+="--seed $SEED "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--save_model "
    cmd+="2>$ERRORS_DIR/errors_expression_${gene}.txt"
    echo "Running: $cmd"
    eval $cmd

done

