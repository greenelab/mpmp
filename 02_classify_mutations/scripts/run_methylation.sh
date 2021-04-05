#!/bin/bash
RESULTS_DIR=./results/bmiq_results
ERRORS_DIR=./bmiq_results_errors
SEED=42
N_DIM=5000

mkdir -p $ERRORS_DIR

for data_type in expression me_27k_bmiq; do
    cmd="python 02_classify_stratified/run_mutation_classification.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $SEED 2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done

# use compressed data for methylation
for data_type in me_27k me_450k; do
    cmd="python 03_classify_compressed/run_mutation_compressed.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $SEED --n_dim $N_DIM 2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done
