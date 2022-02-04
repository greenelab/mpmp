genes="TP53 KRAS EGFR PIK3CA IDH1 SETD2"
num_feats=1000

for gene in $genes; do

    for seed in 42 1; do

        results_dir=./02_classify_mutations/results/compare_nonlinear/linear_${num_feats}
        errors_dir=./linear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--num_features $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/compare_nonlinear/nonlinear_${num_feats}
        errors_dir=./nonlinear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--num_features $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="--nonlinear "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/compare_nonlinear/linear_f_test_${num_feats}
        errors_dir=./linear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--feature_selection f_test "
        cmd+="--num_features $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/compare_nonlinear/nonlinear_f_test_${num_feats}
        errors_dir=./nonlinear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--feature_selection f_test "
        cmd+="--num_features $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="--nonlinear "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
