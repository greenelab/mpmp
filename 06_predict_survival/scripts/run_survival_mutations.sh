#!/bin/bash
ERRORS_DIR=./survival_errors
mkdir -p $ERRORS_DIR

for seed in 42 1; do

    results_dir=./06_predict_survival/results/mutations_me_all/

    # run baseline with clinical covariates only
    cmd="python 06_predict_survival/run_survival_prediction.py "
    cmd+="--results_dir $results_dir "
    cmd+="--seed $seed "
    cmd+="--fit_ridge "
    cmd+="--overlap_data_types mutation expression me_27k me_450k "
    cmd+="--training_data baseline "
    cmd+="2>$ERRORS_DIR/errors_baseline_s${seed}.txt"
    echo "Running: $cmd"
    eval $cmd

    # run prediction with all mutations and significant mutations
    for data_type in vogelstein_mutations significant_mutations; do
        cmd="python 06_predict_survival/run_survival_prediction.py "
        cmd+="--results_dir $results_dir "
        cmd+="--seed $seed "
        cmd+="--fit_ridge "
        cmd+="--overlap_data_types mutation expression me_27k me_450k "
        cmd+="--training_data $data_type "
        cmd+="2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

    # run prediction with top 10 PCs of each -omics type
    n_dim=10
    for data_type in expression me_27k me_450k; do
        cmd="python 06_predict_survival/run_survival_prediction.py "
        cmd+="--n_dim $n_dim "
        cmd+="--results_dir $results_dir "
        cmd+="--seed $seed "
        cmd+="--fit_ridge "
        cmd+="--overlap_data_types mutation expression me_27k me_450k "
        cmd+="--training_data $data_type "
        cmd+="2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
        echo "Running: $cmd"
        eval $cmd
    done

done
