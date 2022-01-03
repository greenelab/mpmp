genes="TP53 KRAS XIST HERC2"
num_feats=10

for gene in $genes; do

    for seed in 42 1; do

        results_dir=./02_classify_mutations/results/shuffle_then_bc/linear_${num_feats}
        errors_dir=./linear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--subset_mad_genes $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/shuffle_then_bc/nonlinear_${num_feats}
        errors_dir=./nonlinear_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--subset_mad_genes $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="--nonlinear "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/shuffle_then_bc/linear_bc_train_test_${num_feats}
        errors_dir=./linear_bc_train_test_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--subset_mad_genes $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="--bc_train_test "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

        results_dir=./02_classify_mutations/results/shuffle_then_bc/nonlinear_bc_train_test_${num_feats}
        errors_dir=./nonlinear_bc_train_test_${num_feats}_errors
        mkdir -p $errors_dir
        cmd="python 02_classify_mutations/run_mutation_classification.py "
        cmd+="--gene_set custom "
        cmd+="--custom_genes $gene "
        cmd+="--overlap_data_types expression "
        cmd+="--results_dir $results_dir "
        cmd+="--subset_mad_genes $num_feats "
        cmd+="--seed $seed "
        cmd+="--training_data expression "
        cmd+="--bc_train_test "
        cmd+="--nonlinear "
        cmd+="2>$errors_dir/errors_expression_$gene_$seed.txt"
        echo "Running: $cmd"
        eval $cmd

    done

done
