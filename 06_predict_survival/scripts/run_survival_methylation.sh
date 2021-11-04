#!/bin/bash
ERRORS_DIR=./survival_errors
mkdir -p $ERRORS_DIR

for seed in 42 1; do

    for n_dim in 10 100 500 1000 5000; do

        # top n_dim PCA features
        results_dir=./06_predict_survival/results/me_ridge/results_${n_dim}_pca/
        for data_type in expression me_27k me_450k; do
            cmd="python 06_predict_survival/run_survival_prediction.py "
            cmd+="--n_dim $n_dim "
            cmd+="--results_dir $results_dir "
            cmd+="--seed $seed "
            cmd+="--overlap_data_types expression me_27k me_450k "
            cmd+="--training_data ${data_type} "
            cmd+="2>$ERRORS_DIR/errors_${data_type}_s${seed}.txt"
            echo "Running: $cmd"
            eval $cmd
        done

    done

done
