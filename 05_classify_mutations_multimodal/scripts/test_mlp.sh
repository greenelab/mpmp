#!/bin/bash
RESULTS_DIR=./05_classify_mutations_multimodal/results/test_mlp
ERRORS_DIR=./mlp_pilot_errors
GENES="KRAS EGFR PIK3CA SETD2"

n_dim=1000

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # run all combinations of expression, me_27k, me_450k
    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k "
    cmd+="--num_features -1 "
    cmd+="--n_dim $n_dim $n_dim "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--model mlp "
    cmd+="2>$ERRORS_DIR/errors_expression_me_27k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_450k "
    cmd+="--num_features -1 "
    cmd+="--n_dim $n_dim $n_dim "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--model mlp "
    cmd+="2>$ERRORS_DIR/errors_expression_me_450k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k me_450k "
    cmd+="--num_features -1 "
    cmd+="--n_dim $n_dim $n_dim "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--model mlp "
    cmd+="2>$ERRORS_DIR/errors_me_27k_me_450k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k me_450k "
    cmd+="--num_features -1 "
    cmd+="--n_dim $n_dim $n_dim $n_dim "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--model mlp "
    cmd+="2>$ERRORS_DIR/errors_all.txt"
    echo "Running: $cmd"
    eval $cmd

done
