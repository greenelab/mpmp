#!/usr/bin/env python
# coding: utf-8

# ## Plot mutation prediction results

# In this notebook, we'll visualize the results of our mutation prediction experiments across all data types (see `README.md` for more details). The files analyzed in this notebook are generated by the `run_mutation_prediction.py` script.
# 
# Notebook parameters:
# * SIG_ALPHA (float): significance cutoff (after FDR correction)
# * PLOT_AUROC (bool): if True plot AUROC, else plot AUPR

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


# set results directory
results_dir = Path(cfg.results_dirs['mutation'],
                   'all_data_types_results_shuffle_cancer_type',
                   'gene').resolve()

# set significance cutoff after FDR correction
SIG_ALPHA = 0.001

# if True, save figures to ./images directory
SAVE_FIGS = True

# if True, plot AUROC instead of AUPR
PLOT_AUROC = False
if PLOT_AUROC:
    plot_metric = 'auroc'
    images_dir = Path(cfg.images_dirs['mutation'], 'auroc')
else:
    plot_metric = 'aupr'
    images_dir = Path(cfg.images_dirs['mutation'])


# In[3]:


# load raw data
results_df = au.load_stratified_prediction_results(results_dir, 'gene')

# here we want to use compressed data for methylation datasets (27k and 450k)
# the results in 02_classify_compressed/compressed_vs_raw_results.ipynb show that
# performance is equal or slightly better for PCA compressed methylation data,
# and it's much easier/faster to fit models on
results_df = results_df[results_df.training_data.isin(['expression', 'rppa', 'mirna', 'mut_sigs'])]

# make sure that we're correctly pointing to raw data for non-methylation data types
# and that we have data for two replicates (two random seeds)
print(results_df.shape)
print(results_df.seed.unique())
print(results_df.training_data.unique())
results_df.head()


# In[4]:


# load compressed data for me_27k and me_450k
compressed_results_df = au.load_compressed_prediction_results(results_dir,
                                                              'gene',
                                                              old_filenames=True)

# make sure that we're correctly pointing to compressed methylation data
# and that we have data for one dimension and two replicates (two random seeds)
print(compressed_results_df.shape)
print(compressed_results_df.seed.unique())
print(compressed_results_df.training_data.unique())
print(compressed_results_df.n_dims.unique())
compressed_results_df.head()


# In[5]:


results_df['n_dims'] = 'raw'
results_df = pd.concat((results_df, compressed_results_df))
print(results_df.seed.unique())
print(results_df.training_data.unique())
print(results_df.n_dims.unique())
print(results_df.shape)
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
results_df.training_data.replace(to_replace=training_data_map, inplace=True)
results_df.head()


# In[6]:


all_results_df = au.compare_all_data_types(results_df,
                                           SIG_ALPHA,
                                           metric=plot_metric)

all_results_df.sort_values(by='p_value').head(10)


# In[7]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric=plot_metric,
                          verbose=True)

if SAVE_FIGS:
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'all_vs_shuffled_extended.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_vs_shuffled_extended.png',
                dpi=300, bbox_inches='tight')


# In[8]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['RPPA', 'microRNA', 'mutational signatures']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric=plot_metric,
                          verbose=True)
    
if SAVE_FIGS:
    plt.savefig(images_dir / 'all_vs_shuffled.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_vs_shuffled.png',
                dpi=300, bbox_inches='tight')


# In[9]:


# map gene/training data combinations to accept/reject null
# vs. shuffled baseline
# we want to plot this info on top of -omics comparison
id_to_sig = (all_results_df
  .loc[:, ['gene', 'training_data', 'reject_null']]
  .rename(columns={'reject_null': 'reject_null_baseline'})
)

id_to_sig.head()


# In[10]:


# compare expression against all other data modalities
# could do all vs. all, but that would give us lots of plots
sns.set({'figure.figsize': (16, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_comparison(results_df,
                            axarr,
                            filtered_data_map,
                            SIG_ALPHA,
                            metric=plot_metric,
                            sig_genes=id_to_sig,
                            verbose=True)

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_comparison_extended.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_comparison_extended.png',
                dpi=300, bbox_inches='tight')


# In[11]:


# compare expression against all other data modalities
# could do all vs. all, but that would give us lots of plots
sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

datasets = ['gene expression', 'RPPA', 'microRNA', 'mutational signatures']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

fig, axarr = plt.subplots(1, 3)

plu.plot_volcano_comparison(results_df,
                            axarr,
                            filtered_data_map,
                            SIG_ALPHA,
                            metric=plot_metric,
                            sig_genes=id_to_sig,
                            verbose=True)

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_comparison.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_comparison.png',
                dpi=300, bbox_inches='tight')


# In[12]:


sns.set({'figure.figsize': (12, 9)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(2, 1)

tests_df = plu.plot_boxes(all_results_df,
                          axarr,
                          training_data_map,
                          metric=plot_metric,
                          orientation='v',
                          verbose=True,
                          pairwise_tests=True,
                          pairwise_box_pairs=[('gene expression', '27k methylation'),
                                              ('27k methylation', '450k methylation'),
                                              ('450k methylation', 'RPPA'),
                                              ('RPPA', 'microRNA'),
                                              ('microRNA', 'mutational signatures')])

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_boxes.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_boxes.png',
                dpi=300, bbox_inches='tight')


# In[13]:


# pairwise rank sum tests comparing results distributions
# H0: results distributions are the same between the data types
tests_df.sort_values(['gene_set', 'p_value'])


# In[14]:


sns.set({'figure.figsize': (15, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

plu.plot_boxes(all_results_df,
               axarr,
               training_data_map,
               metric=plot_metric,
               verbose=True)


# In[15]:


heatmap_df = (all_results_df
    .pivot(index='training_data', columns='gene', values='delta_mean')
    .reindex(training_data_map.values())
)
heatmap_df.iloc[:, :5]


# In[16]:


sns.set({'figure.figsize': (28, 6)})
sns.set_context('notebook', font_scale=1.5)

ax = plu.plot_heatmap(heatmap_df,
                      all_results_df.reset_index(drop=True),
                      results_df,
                      metric=plot_metric,
                      origin_eps_x=0.02,
                      origin_eps_y=0.015,
                      length_x=0.85,
                      length_y=0.95)

plt.title('Performance by data type for Vogelstein et al. genes, all data types', pad=15)

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_heatmap.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_heatmap.png',
                dpi=300, bbox_inches='tight')


# Key to above heatmap:
# 
# * A grey dot = significantly better than label-permuted baseline, but significantly worse than best-performing data type
# * A grey dot with black dot inside = significantly better than label-permuted baseline, and not significantly different from best-performing data type (i.e. "statistically equivalent to best")
# * No dot = not significantly better than label-permuted baseline
