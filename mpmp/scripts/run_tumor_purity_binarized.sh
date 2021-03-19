#!/bin/bash
# Script to run binarized tumor purity classification experiments, across
# different data types and compressed/uncompressed data sets.
RESULTS_DIR=./results/purity_binarized_new
ERRORS_DIR=./purity_binarized_errors
N_DIM=5000

mkdir -p $ERRORS_DIR

for seed in 42 1; do
    for data_type in expression me_27k; do
        cmd="python 02_classify_stratified/run_purity_prediction.py --results_dir $RESULTS_DIR --training_data ${data_type} --seed $seed --classify --output_preds 2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done
    # use compressed data for 450K methylation
    for data_type in expression me_27k me_450k; do
        cmd="python 02_classify_stratified/run_purity_prediction.py --results_dir $RESULTS_DIR --training_data ${data_type} --seed $seed --use_compressed --classify --output_preds 2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done
done
