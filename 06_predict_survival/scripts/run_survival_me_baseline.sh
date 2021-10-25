#!/bin/bash
ERRORS_DIR=./survival_errors
mkdir -p $ERRORS_DIR

for seed in 42 1; do

    results_dir=./06_predict_survival/results/me_ridge_baseline/
    cmd="python 06_predict_survival/run_survival_prediction.py "
    cmd+="--results_dir $results_dir "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--training_data baseline "
    cmd+="2>$ERRORS_DIR/errors_baseline_s${seed}.txt"
    echo "Running: $cmd"
    eval $cmd

done
