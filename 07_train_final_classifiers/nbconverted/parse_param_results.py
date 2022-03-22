#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# experiment results directory containing parameter lists
results_dir = Path(
    cfg.results_dirs['mutation'],
    'merged_filter_all',
    'gene'
).resolve()


# In[28]:


data_type_options = [
    'expression',
    'me_27k',
    'me_450k',
    'rppa',
    'mirna',
    'mut_sigs'
]

def get_data_type(results_filename):
    # this is a bit messy but it works
    for data_type in data_type_options:
        if data_type in results_filename:
            return data_type
    return None

def get_best_params(results_file):
    results_df = pd.read_csv(results_file, sep='\t', index_col=0)
    param_names = None 
    params, metrics = [], []
    for fold_no in results_df.fold.unique():
        fold_results_df = (
            results_df[results_df.fold == fold_no]
              .sort_values(by='mean_test_score', ascending=False)
        )
        if param_names is None:
            non_param_cols = ['fold', 'mean_train_score', 'mean_test_score']
            param_names = [
                c for c in fold_results_df.columns
                  if c not in non_param_cols
            ]
        params.append(
            fold_results_df.head(1).loc[:, param_names].values[0].tolist()
        )
        metrics.append(
            fold_results_df.head(1).loc[:, 'mean_test_score'].values[0]
        )
    return param_names, params, metrics

def get_best_params_folds(results_dir, gene):
    all_best_params = {dt: [] for dt in data_type_options}
    gene_results_dir = results_dir / gene
    for results_file in gene_results_dir.iterdir():
        if not results_file.is_file():
            continue
        results_filename = str(results_file.stem)
        if ('param_grid' not in results_filename or
            'signal'not in results_filename):
            continue
        # parse filename
        data_type = get_data_type(results_filename)
        if '_n5000' in results_filename:
            seed = int(results_filename
                .split('_')[-4]
                .replace('s', '')
            )
        else:
            seed = int(results_filename
                .split('_')[-3]
                .replace('s', '')
            )
        # fold = ...
        param_names, params, metrics = get_best_params(results_file)
        all_best_params[data_type].append(
            (seed, param_names, params, metrics)
        )
    return all_best_params

print(
    get_best_params_folds(results_dir, 'BRAF')
)

