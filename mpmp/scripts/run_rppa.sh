#!/bin/bash
RESULTS_DIR=./results/rppa_results
ERRORS_DIR=./rppa_results_errors
SEED=42

mkdir -p $ERRORS_DIR

for data_type in rppa expression me_27k me_450k; do
    cmd="python 01_classify_stratified/run_mutation_classification.py --gene_set vogelstein --results_dir $RESULTS_DIR --training_data ${data_type} --seed $SEED 2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done
