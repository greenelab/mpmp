#!/bin/bash
RESULTS_DIR=./02_classify_mutations/results/methylation_results
ERRORS_DIR=./methylation_errors

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # use raw data for expression
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set vogelstein "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd

    # use compressed data for methylation
    for data_type in me_27k me_450k; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data ${data_type} "
        cmd+="--seed $seed "
        cmd+="--n_dim $N_DIM "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
