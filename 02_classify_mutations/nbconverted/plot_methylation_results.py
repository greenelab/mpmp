#!/usr/bin/env python
# coding: utf-8

# ## Plot mutation prediction results

# In this notebook, we'll compare the results of our mutation prediction experiments for expression and methylation data only (see `README.md` for more details). The files analyzed in this notebook are generated by the `run_mutation_prediction.py` script.
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
from adjustText import adjust_text

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.plot_utilities as plu


# In[2]:


# set results directory
results_dir = Path(cfg.results_dirs['mutation'],
                   'methylation_results',
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
results_df = results_df[results_df.training_data.isin(['expression'])].copy()

print(results_df.shape)
print(results_df.seed.unique())
print(results_df.training_data.unique())
results_df.head()


# In[4]:


# load compressed data for me_27k and me_450k
compressed_results_df = au.load_compressed_prediction_results(results_dir, 'gene',
                                                              old_filenames=True)
compressed_results_df = compressed_results_df[
    (compressed_results_df.training_data.isin(['me_27k', 'me_450k'])) &
    (compressed_results_df.n_dims == 5000)
].copy()
print(compressed_results_df.seed.unique())
print(compressed_results_df.training_data.unique())
print(compressed_results_df.n_dims.unique())
print(compressed_results_df.shape)
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
}
results_df.training_data.replace(to_replace=training_data_map, inplace=True)
results_df.head()


# In[6]:


all_results_df = au.compare_all_data_types(results_df,
                                           SIG_ALPHA,
                                           metric=plot_metric)

all_results_df.sort_values(by='p_value').head(10)


# In[7]:


sns.set({'figure.figsize': (24, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

plu.plot_volcano_baseline(all_results_df,
                          axarr,
                          training_data_map,
                          SIG_ALPHA,
                          metric=plot_metric,
                          verbose=True)
    
if SAVE_FIGS:
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'methylation_vs_shuffled.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'methylation_vs_shuffled.png',
                dpi=300, bbox_inches='tight')


# In[8]:


# map gene/training data combinations to accept/reject null
# vs. shuffled baseline
# we want to plot this info on top of -omics comparison
id_to_sig = (all_results_df
  .loc[:, ['gene', 'training_data', 'reject_null']]
  .rename(columns={'reject_null': 'reject_null_baseline'})
)

id_to_sig.head()


# In[9]:


# compare expression against all other data modalities
# could do all vs. all, but that would give us lots of plots
sns.set({'figure.figsize': (16, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)


plu.plot_volcano_comparison(results_df,
                            axarr,
                            training_data_map,
                            SIG_ALPHA,
                            metric=plot_metric,
                            xlim=(-0.6, 0.6),
                            sig_genes=id_to_sig,
                            verbose=True)

if SAVE_FIGS:
    plt.savefig(images_dir / 'methylation_comparison.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'methylation_comparison.png',
                dpi=300, bbox_inches='tight')


# In[10]:


sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

tests_df = plu.plot_boxes(all_results_df,
                          axarr,
                          training_data_map,
                          metric=plot_metric,
                          verbose=True,
                          pairwise_tests=True,
                          pairwise_box_pairs=[('gene expression', '27k methylation'),
                                              ('27k methylation', '450k methylation'),
                                              ('gene expression', '450k methylation')])

if SAVE_FIGS:
    plt.savefig(images_dir / 'methylation_best_boxes.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'methylation_best_boxes.png',
                dpi=300, bbox_inches='tight')


# In[11]:


# pairwise rank sum tests comparing results distributions
# H0: results distributions are the same between the data types
tests_df.sort_values(['gene_set', 'p_value'])


# In[12]:


heatmap_df = (all_results_df
    .pivot(index='training_data', columns='gene', values='delta_mean')
    .reindex(training_data_map.values())
)
heatmap_df.iloc[:, :5]


# In[13]:


sns.set({'figure.figsize': (32, 5)})
sns.set_context('notebook', font_scale=1.5)

ax = plu.plot_heatmap(heatmap_df,
                      all_results_df.reset_index(drop=True),
                      different_from_best=True,
                      raw_results_df=results_df,
                      metric=plot_metric)

plt.title('Performance by data type for Vogelstein et al. genes, expression vs. methylation', pad=15)

if SAVE_FIGS:
    plt.savefig(images_dir / 'methylation_heatmap.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'methylation_heatmap.png',
                dpi=300, bbox_inches='tight')

