#!/bin/bash
RESULTS_DIR=./results/mut_sigs_results
ERRORS_DIR=./mut_sigs_results_errors
SEED=1
N_DIM=5000

mkdir -p $ERRORS_DIR

for data_type in mut_sigs rppa expression; do
    cmd="python 01_classify_stratified/run_mutation_classification.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $SEED 2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done

# use compressed data for methylation
for data_type in me_27k me_450k; do
    cmd="python 02_classify_compressed/run_mutation_compressed.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $SEED --n_dim $N_DIM 2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done
