#!/bin/bash
RESULTS_DIR=./02_classify_mutations/results/methylation_results
ERRORS_DIR=./methylation_errors

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # run raw data for expression
    # note all the raw datasets here default to 8,000 features (chosen by MAD)
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set vogelstein "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_expression_raw.txt"
    echo "Running: $cmd"
    eval $cmd

    # run compressed data for expression
    for n_dim in 100 1000 5000; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data expression "
        cmd+="--seed $seed "
        cmd+="--n_dim $n_dim "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="2>$ERRORS_DIR/errors_expression_${n_dim}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

    # run raw data for 27k methylation
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set vogelstein "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_me_27k_raw.txt"
    echo "Running: $cmd"
    eval $cmd

    # run compressed data for 27k methylation
    for n_dim in 100 1000 5000; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data me_27k "
        cmd+="--seed $seed "
        cmd+="--n_dim $n_dim "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="2>$ERRORS_DIR/errors_me_27k_${n_dim}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

    # run raw data for 450k methylation
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set vogelstein "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_450k "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_me_450k_raw.txt"
    echo "Running: $cmd"
    eval $cmd

    # run compressed data for 450k methylation
    for n_dim in 100 1000 5000; do
        cmd="python 02_classify_mutations/run_mutation_compressed.py "
        cmd+="--gene_set vogelstein "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--training_data me_450k "
        cmd+="--seed $seed "
        cmd+="--n_dim $n_dim "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="2>$ERRORS_DIR/errors_me_450k_${n_dim}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
