#!/bin/bash
PARAMS_DIR=./02_classify_mutations/results/merged_all_params
RESULTS_DIR=./07_train_final_classifiers/results/merged_all_params
ERRORS_DIR=./final_merged_errors
SEED=42
N_DIM=5000

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

    # use raw data for non-methylation data types
    for training_data in expression rppa mirna mut_sigs; do
        cmd="python 07_train_final_classifiers/train_classifier.py "
        cmd+="--gene $gene "
        cmd+="--params_dir $PARAMS_DIR "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data ${training_data} "
        cmd+="--seed $SEED "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        cmd+="--save_model "
        cmd+="2>$ERRORS_DIR/errors_${training_data}_${gene}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

    # use compressed data for methylation
    for training_data in me_27k me_450k; do
        cmd="python 07_train_final_classifiers/train_classifier.py "
        cmd+="--gene $gene "
        cmd+="--params_dir $PARAMS_DIR "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data ${training_data} "
        cmd+="--seed $SEED "
        cmd+="--n_dim $N_DIM "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        cmd+="--save_model "
        cmd+="2>$ERRORS_DIR/errors_${training_data}_${gene}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done

