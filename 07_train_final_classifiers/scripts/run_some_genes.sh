#!/bin/bash
PARAMS_DIR=./02_classify_mutations/results/merged_filter_all
RESULTS_DIR=./07_train_final_classifiers/results/pilot_genes
ERRORS_DIR=./final_pilot_errors
GENES=(
  "TP53"
  "KRAS"
  "EGFR"
  "PIK3CA"
  "IDH1"
  "SETD2"
)
SEED=42

mkdir -p $ERRORS_DIR

for gene in "${GENES[@]}"; do

    # just do expression for now
    # TODO: could do methylation but would need to pass PCA features
    cmd="python 07_train_final_classifiers/train_classifier.py "
    cmd+="--gene $gene "
    cmd+="--params_dir $PARAMS_DIR "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression "
    cmd+="--seed $SEED "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--save_model "
    cmd+="2>$ERRORS_DIR/errors_expression.txt"
    echo "Running: $cmd"
    eval $cmd

done

