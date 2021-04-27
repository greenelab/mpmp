#!/usr/bin/env python
# coding: utf-8

# ## Calculate "missing genes" for all experiments
# 
# We observed that as we added more data types, we're ending up with fewer genes in our cross-validation experiments. The reasons for this are described in more detail in https://github.com/greenelab/mpmp/issues/44, but basic reasons include:
# 
# * No valid cancer types that meet our filters (this is more likely with more data types -> fewer samples)
# * In one or more cross-validation splits, either the train or test set has no positive or negative samples (i.e. only one class -> AUROC/AUPR are not defined)
# 
# In this script, we want to calculate the number of missing genes without having to fit every model (this will run on the order of ~1 hour, rather than a day or two with the model fits).

# In[1]:


import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.prediction.cross_validation as cv
import mpmp.utilities.data_utilities as du


# In[2]:


def calculate_gene_count(overlap_data_types, seeds, num_folds):
    """For a set of data types, calculate the number of valid genes."""
    gene_seed_list = []
    sample_info_df = du.load_sample_info('expression')
    for seed in seeds:
        tcga_data = TCGADataModel(seed=seed,
                                  overlap_data_types=overlap_data_types)
        genes_df = tcga_data.load_gene_set('vogelstein')
        for gene_ix, gene_series in genes_df.iterrows():
            
            print(gene_series.gene, file=sys.stderr)
            try:
                tcga_data.process_data_for_gene(gene_series.gene,
                                                gene_series.classification,
                                                None)
            except KeyError: continue
            y_ones = np.count_nonzero(tcga_data.y_df.status)
            y_zeroes = len(tcga_data.y_df.status) - y_ones
            print(y_ones, y_zeroes, file=sys.stderr)
            
            # check if any valid cancer types, if not break
            if tcga_data.X_df.shape[0] == 0:
                gene_seed_list.append((gene_series.gene, seed, False, 'no_valid_cancer_types'))
                continue
                
            # subset features to speed up CV
            tcga_data.X_df = tcga_data.X_df.iloc[:, :50]
                
            # if valid cancer types, look at CV folds and make sure each
            # has 0 and 1 labels
            gene_seed_valid = True
            reason = 'N/A'
            for fold_no in range(num_folds):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            message='The least populated class in y')
                    X_train, X_test, _ = cv.split_stratified(
                        tcga_data.X_df,
                        sample_info_df,
                        num_folds=num_folds,
                        fold_no=fold_no,
                        seed=seed
                    )
                y_train = tcga_data.y_df.reindex(X_train.index)
                y_test = tcga_data.y_df.reindex(X_test.index)
                
                # count 0/1 labels in y_train and y_test
                y_train_ones = np.count_nonzero(y_train.status)
                y_train_zeroes = len(y_train.status) - y_train_ones
                y_test_ones = np.count_nonzero(y_test.status)
                y_test_zeroes = len(y_test.status) - y_test_ones
                print(fold_no, y_train_ones, y_train_zeroes, y_test_ones, y_test_zeroes,
                      file=sys.stderr)
                
                if ((y_train_ones == 0) or (y_train_zeroes == 0)):
                    gene_seed_valid = False
                    reason = 'one_train_class'
                    break
                elif ((y_test_ones == 0) or (y_test_zeroes == 0)):
                    gene_seed_valid = False
                    reason = 'one_test_class'
                    break
                    
            gene_seed_list.append((gene_series.gene, seed, gene_seed_valid, reason))
                
    return gene_seed_list


# In[3]:


training_data_types = {
    'expression': ['expression'],
    'methylation': ['expression', 'me_27k', 'me_450k'],
    'all': ['expression', 'me_27k', 'me_450k', 'rppa', 'mirna', 'mut_sigs']
}
seeds = [42, 1]
num_folds = 4

gene_counts_dir = Path('./', 'gene_counts')
gene_counts_dir.mkdir(exist_ok=True)

# run for all the data types ("experiments") we want to consider
# if results already exist we don't have to recalculate them
for dataset, overlap_data_types in training_data_types.items():
    gene_count_file = gene_counts_dir / 'gene_count_{}.tsv'.format(dataset)
    if gene_count_file.is_file():
        print('File {} exists'.format(gene_count_file), file=sys.stderr)
    else:
        print('File {} does not exist, calculating'.format(
            gene_count_file), file=sys.stderr)
        lst = calculate_gene_count(overlap_data_types, seeds, num_folds)
        with open (gene_count_file, 'w') as f:
            for t in lst:
                f.write('\t'.join([str(v) for v in t]))
                f.write('\n')


# In[4]:


# now load gene count files and count valid genes
for dataset in training_data_types.keys():
    gene_counts_df = pd.read_csv(
        gene_counts_dir / 'gene_count_{}.tsv'.format(dataset),
        names=['gene', 'seed', 'is_valid', 'reason'], 
        sep='\t'
    )
    valid_genes = gene_counts_df.groupby('gene').all().is_valid
    print('{}:'.format(dataset),
          '{} valid /'.format(np.count_nonzero(valid_genes)),
          '{} total'.format(len(valid_genes)))

