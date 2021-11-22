#!/bin/bash
ERRORS_DIR=./predictions_errors
SEED=42

mkdir -p $ERRORS_DIR

for data_type in expression me_27k me_450k; do
    cmd="python 02_classify_mutations/predict_all_genes.py "
    cmd+="--training_data $data_type "
    cmd+="--seed $SEED "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_${data_type}.txt"
    echo "Running: $cmd"
    eval $cmd
done
