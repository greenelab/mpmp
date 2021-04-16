#!/bin/bash
RESULTS_DIR=./results/all_data_types_results
ERRORS_DIR=./all_data_types_errors
N_DIM=5000

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # use raw data for non-expression data types
    for data_type in mirna mut_sigs rppa expression; do
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data ${data_type} "
        cmd+="--seed $seed "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

    # use compressed data for methylation
    for data_type in me_27k me_450k; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data ${data_type} "
        cmd+="--seed $SEED "
        cmd+="--n_dim $N_DIM "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
