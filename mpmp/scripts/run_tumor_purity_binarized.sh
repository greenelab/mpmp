#!/bin/bash
RESULTS_DIR=./results/purity_binarized_results
ERRORS_DIR=./purity_binarized_errors
N_DIM=5000

mkdir -p $ERRORS_DIR

for seed in 42 1; do
    for data_type in expression me_27k; do
        cmd="python 02_classify_stratified/run_purity_prediction.py --results_dir $RESULTS_DIR --training_data ${data_type} --seed $seed 2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done
    # use compressed data for 450K methylation
    cmd="python 02_classify_stratified/run_purity_prediction.py --results_dir $RESULTS_DIR --training_data me_450k --seed $seed --use_compressed 2>$ERRORS_DIR/errors_me_450k_s${seed}.txt"
    echo "Running: $cmd"
    eval $cmd
done
