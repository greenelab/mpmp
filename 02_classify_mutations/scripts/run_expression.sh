#!/bin/bash

# Run mutation classification experiments for expression data,
# using various gene sets

RESULTS_DIR=./02_classify_mutations/results/50_random
ERRORS_DIR=./50_random_errors

mkdir -p $ERRORS_DIR

for seed in 42 1; do
    cmd="python 02_classify_mutations/run_mutation_classification.py --gene_set 50_random --results_dir $RESULTS_DIR --training_data expression --seed $seed 2>$ERRORS_DIR/errors_expression_s$seed.txt"
    echo "Running: $cmd"
    eval $cmd
done
