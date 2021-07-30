#!/bin/bash
N_FEATURES=1000
RESULTS_DIR=./06_predict_survival/results/results_${N_FEATURES}_top_mad/
ERRORS_DIR=./survival_errors

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # use raw data for non-methylation data types
    for data_type in expression me_27k me_450k; do
        cmd="python 06_predict_survival/run_survival_prediction.py "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--seed $seed "
        cmd+="--subset_mad_genes $N_FEATURES "
        cmd+="--overlap_data_types expression me_27k me_450k "
        cmd+="--training_data ${data_type} "
        cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
