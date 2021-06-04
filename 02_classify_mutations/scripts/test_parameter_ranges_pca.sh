RESULTS_DIR=./02_classify_mutations/results/params/single_omics_pca
ERRORS_DIR=./params_errors

mkdir -p $ERRORS_DIR

for seed in 42 1; do

    cmd="python 02_classify_mutations/run_mutation_compressed.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes EGFR PIK3CA KRAS TP53 "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data expression "
    cmd+="--seed $seed "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--n_dim 5000 "
    cmd+="--save_inner_cv_details "
    cmd+="2>$ERRORS_DIR/errors_expression_${seed}.txt"
    echo "Running: $cmd"
    eval $cmd

    cmd="python 02_classify_mutations/run_mutation_compressed.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes EGFR PIK3CA KRAS TP53 "
    cmd+="--results_dir $RESULTS_DIR "
    cmd+="--training_data me_27k "
    cmd+="--seed $seed "
    cmd+="--n_dim 5000 "
    cmd+="--overlap_data_types expression me_27k me_450k "
    cmd+="--save_inner_cv_details "
    cmd+="2>$ERRORS_DIR/errors_me_27k_${seed}.txt"
    echo "Running: $cmd"
    eval $cmd

done
