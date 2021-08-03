#!/bin/bash
ERRORS_DIR=./msi_errors
N_DIM=5000
RESULTS_DIR=./04_predict_controls/results/msi/msi_${N_DIM}_top_mad

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # use raw data for non-methylation data types
    for data_type in mirna mut_sigs rppa expression me_27k me_450k; do
        cmd="python 04_predict_controls/run_msi_prediction.py "
        cmd+="--results_dir $RESULTS_DIR "
        cmd+="--overlap_data_types expression me_27k me_450k rppa mirna mut_sigs "
        cmd+="--training_data ${data_type} "
        cmd+="--seed $seed "
        cmd+="--subset_mad_genes $N_DIM "
        cmd+="2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
