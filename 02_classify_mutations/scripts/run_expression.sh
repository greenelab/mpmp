#!/bin/bash

# Run mutation classification experiments for expression data,
# using various gene sets
for dataset in top_50 50_random vogelstein; do
    results_dir=./02_classify_mutations/results/${dataset}_expression_only
    errors_dir=./${dataset}_errors
    mkdir -p $errors_dir
    for seed in 42 1; do
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set $dataset "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="2>$errors_dir/errors_expression_s$seed.txt"
        echo "Running: $cmd"
        eval $cmd
    done
done
