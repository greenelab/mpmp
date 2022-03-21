#!/usr/bin/env python
# coding: utf-8

# ## Compare dataset filtering methods
# 
# TODO: explain

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


orig_results_dir = Path(cfg.results_dirs['mutation'],
                        'merged_all',
                        'gene').resolve()

new_results_dir = Path(cfg.results_dirs['mutation'],
                       'merged_filter_all',
                       'gene').resolve()

# set significance cutoff after FDR correction
SIG_ALPHA = 0.001


# In[3]:


# load raw data
new_results_df = au.load_stratified_prediction_results(new_results_dir, 'gene')

# here we want to use compressed data for methylation datasets (27k and 450k)
# the results in 02_classify_compressed/compressed_vs_raw_results.ipynb show that
# performance is equal or slightly better for PCA compressed methylation data,
# and it's much easier/faster to fit models on
new_results_df = new_results_df[new_results_df.training_data.isin(['expression', 'rppa', 'mirna', 'mut_sigs'])]

# make sure that we're correctly pointing to raw data for non-methylation data types
# and that we have data for two replicates (two random seeds)
print(new_results_df.shape)
print(new_results_df.seed.unique())
print(new_results_df.training_data.unique())
new_results_df.head()


# In[4]:


# load compressed data for me_27k and me_450k
new_compressed_results_df = au.load_compressed_prediction_results(
    new_results_dir, 'gene')

# make sure that we're correctly pointing to compressed methylation data
# and that we have data for one dimension and two replicates (two random seeds)
print(new_compressed_results_df.shape)
print(new_compressed_results_df.seed.unique())
print(new_compressed_results_df.training_data.unique())
print(new_compressed_results_df.n_dims.unique())
new_compressed_results_df.head()


# In[5]:


new_results_df['n_dims'] = 'raw'
new_results_df = pd.concat((new_results_df, new_compressed_results_df))
print(new_results_df.seed.unique())
print(new_results_df.training_data.unique())
print(new_results_df.n_dims.unique())
print(new_results_df.shape)
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
new_results_df.training_data.replace(to_replace=training_data_map, inplace=True)
new_results_df.head()


# In[7]:


new_all_results_df = au.compare_all_data_types(new_results_df,
                                               SIG_ALPHA,
                                               metric='aupr')

new_all_results_df.sort_values(by='p_value').head(10)


# In[8]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(new_all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)


# In[11]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['RPPA', 'microRNA', 'mutational signatures']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(new_all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)

