#!/bin/bash
RESULTS_DIR=./results/compressed_results
ERRORS_DIR=./compressed_results_errors
SEED=1

mkdir -p $ERRORS_DIR


for seed in 42 1; do
    # run gene expression with 8000 MAD features
    cmd="python 02_classify_mutations/run_mutation_classification.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data expression --seed $seed 2>$ERRORS_DIR/compressed_results_errors_expression_uncompressed.txt"
    echo "Running: $cmd"
    eval $cmd
    # run me_27k and me_450k with all features
    for data_type in me_27k me_450k; do
        cmd="python 02_classify_mutations/run_mutation_classification.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $seed --subset_mad_genes -1 2>$ERRORS_DIR/compressed_results_errors_${data_type}_uncompressed.txt"
        echo "Running: $cmd"
        eval $cmd
    done
done

for seed in 42 1; do
    for n_dim in 100 1000 5000; do
        for data_type in me_450k me_27k expression; do
            cmd="python 02_classify_mutations/run_mutation_compressed.py --gene_set vogelstein --results_dir $RESULTS_DIR --n_dim ${n_dim} --training_data ${data_type} --seed $seed 2>$ERRORS_DIR/compressed_results_errors_${data_type}_${n_dim}.txt"
            echo "Running: $cmd"
            eval $cmd
        done
    done
done
