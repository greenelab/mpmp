#!/bin/bash
ERRORS_DIR=./survival_errors
mkdir -p $ERRORS_DIR

for seed in 42 1; do

    results_dir=./06_predict_survival/results/mutations_me_all/

    # run prediction with mutation predictions
    for data_type in mutation_preds_expression mutation_preds_me_27k mutation_preds_me_450k; do
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

done
