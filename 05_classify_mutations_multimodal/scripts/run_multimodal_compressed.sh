#!/bin/bash
RESULTS_DIR=./05_classify_mutations_multimodal/results/compressed_shuffle_cancer_type
ERRORS_DIR=./multimodal_pilot_errors
GENES="TP53 KRAS EGFR PIK3CA IDH1 SETD2"

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    # run all combinations of expression, me_27k, me_450k
    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_expression_me_27k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_expression_me_450k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_me_27k_me_450k.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 05_classify_mutations_multimodal/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${GENES} "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression me_27k me_450k "
    cmd+="--subset_mad_genes -1 "
    cmd+="--n_dim 5000 5000 5000 "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="2>$ERRORS_DIR/errors_all.txt"
    echo "Running: $cmd"
    eval $cmd

done
