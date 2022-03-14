#!/usr/bin/env python
# coding: utf-8

# ## Plot results of multi-omics classification experiments
# 
# In these experiments, we compare elastic net logistic regression models using multiple data types to models using only a single data type. We're not doing anything particularly fancy here, just concatenating the feature sets (genes or CpG probes) from the individual data types to create a "multi-omics" model.
# 
# For now, we're just doing this for gene expression and the two methylation datasets.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = Path(
    cfg.results_dirs['multimodal'],
    'compressed_shuffle_cancer_type',
    'gene'
).resolve()

# if True, save figures to ./images directory
SAVE_FIGS = True

# if True, plot AUROC instead of AUPR
PLOT_AUROC = False
if PLOT_AUROC:
    plot_metric = 'auroc'
    images_dir = Path(cfg.images_dirs['multimodal'], 'auroc')
else:
    plot_metric = 'aupr'
    images_dir = Path(cfg.images_dirs['multimodal'])


# ## Results with compressed features (figures in main paper)
# 
# We'll also look at results with raw features later, those figures go in the supplement.
# 
# ### Compare raw results

# In[3]:


# load raw data
results_df = au.load_stratified_prediction_results(results_dir, 'gene')

# drop TET2 for now
results_df = results_df[~(results_df.identifier == 'TET2')].copy()

# make sure that we have data for all data types and for two replicates (random seeds)
print(results_df.shape)
print(results_df.seed.unique())
print(results_df.identifier.unique())
print(results_df.training_data.unique())
results_df.head()


# In[4]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (24, 12)})
fig, axarr = plt.subplots(2, 3)
results_df.sort_values(by=['identifier', 'signal', 'training_data'], inplace=True)

data_order =['expression.me_27k',
             'expression.me_450k',
             'me_27k.me_450k',
             'expression.me_27k.me_450k']

plu.plot_multi_omics_raw_results(results_df,
                                 axarr,
                                 data_order,
                                 metric=plot_metric)

handles = []
for ix, data in enumerate(data_order):
    handle = mpatches.Patch(color=sns.color_palette()[ix], label=data)
    handles.append(handle)

axarr[1, 2].legend(handles=handles, loc='lower right')


# ### Compare single-omics and multi-omics results

# In[5]:


# get results from unimodal prediction (individual data types) to compare with
unimodal_results_dir = Path(
    cfg.results_dirs['mutation'],
    'methylation_results_shuffle_cancer_type',
    'gene'
)

# load expression and me_27k results
u_results_df = au.load_compressed_prediction_results(unimodal_results_dir, 'gene')
u_results_df = u_results_df[(u_results_df.n_dims == 5000)].copy()
u_results_df.drop(columns='n_dims', inplace=True)

# make sure data loaded matches our expectations
print(u_results_df.training_data.unique())
print(u_results_df.seed.unique())


# In[6]:


# first, concatenate the unimodal results and the multimodal results
all_results_df = pd.concat((results_df, u_results_df))

print(all_results_df.shape)
print(all_results_df.training_data.unique())
all_results_df.head()


# In[7]:


# then, for each training data type, get the AUPR difference between signal and shuffled
compare_df = pd.DataFrame()
for training_data in all_results_df.training_data.unique():
    data_compare_df = au.compare_control_ind(
        all_results_df[all_results_df.training_data == training_data],
        identifier='identifier',
        metric=plot_metric,
        verbose=True
    )
    data_compare_df['training_data'] = training_data
    data_compare_df.rename(columns={'identifier': 'gene'}, inplace=True)
    compare_df = pd.concat((compare_df, data_compare_df))
    
compare_df.head(10)


# In[8]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (20, 12)})
sns.set_style('whitegrid')

# fig, axarr = plt.subplots(2, 4)
fig, axarr = plt.subplots(2, 3)
compare_df = (
    compare_df[compare_df.gene.isin(results_df.identifier.unique())]
    .sort_values(by=['gene', 'training_data'])
).copy()

data_names = {
    'expression': 'gene expression',
    'me_27k': '27K methylation',
    'me_450k': '450K methylation',
    'expression.me_27k': 'expression + 27K methylation',
    'expression.me_450k': 'expression + 450K methylation',
    'me_27k.me_450k': '27K methylation + 450K methylation',
    'expression.me_27k.me_450k': 'expression + 27K methylation + 450K methylation'
}

# we want to use the same colors as other figures for the individual
# data types, but different colors for the multi-omics models
colors = sns.color_palette()[:3] + sns.color_palette('Dark2')[:4]

plu.plot_multi_omics_results(compare_df,
                             axarr,
                             data_names,
                             colors,
                             metric=plot_metric)

handles = []
for ix, data in enumerate(list(data_names.values())):
    handle = mpatches.Patch(color=colors[ix], label=data)
    handles.append(handle)
    
plt.legend(title='Data types used to train model', handles=handles, loc='lower right')
plt.tight_layout()

if SAVE_FIGS:
    svg_filename = 'multi_omics_boxes.svg'
    png_filename = 'multi_omics_boxes.png'
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / svg_filename, bbox_inches='tight')
    plt.savefig(images_dir / png_filename, dpi=300, bbox_inches='tight')


# ### Compare best-performing single-omics and multi-omics data types

# In[9]:


# for each data type, classify it as single-omics or multi-omics
compare_df['model_type'] = 'Best single-omics'
# multi-omics data types are concatenated using dots
compare_df.loc[compare_df.training_data.str.contains('\.'), 'model_type'] = 'Best multi-omics'
print(compare_df.training_data.unique())
compare_df[compare_df.gene == 'TP53'].head(10)


# In[10]:


sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')

plu.plot_best_multi_omics_results(compare_df,
                                  ylim=(0.0, 0.6),
                                  metric=plot_metric)

if SAVE_FIGS:
    svg_filename = 'multi_omics_best_model.svg'
    png_filename = 'multi_omics_best_model.png'
    plt.savefig(images_dir / svg_filename, bbox_inches='tight')
    plt.savefig(images_dir / png_filename, dpi=300, bbox_inches='tight')


# ## Results with raw features (figures in supplement)
# 
# ### Compare raw results

# In[11]:


results_dir = Path(
    cfg.results_dirs['multimodal'],
    'raw_shuffle_cancer_type',
    'gene'
).resolve()


# In[12]:


# load raw data
results_df = au.load_stratified_prediction_results(results_dir, 'gene')

# drop TET2 for now
results_df = results_df[~(results_df.identifier == 'TET2')].copy()

# make sure that we have data for all data types and for two replicates (random seeds)
print(results_df.shape)
print(results_df.seed.unique())
print(results_df.identifier.unique())
print(results_df.training_data.unique())
results_df.head()


# In[13]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (24, 12)})
fig, axarr = plt.subplots(2, 3)
results_df.sort_values(by=['identifier', 'signal', 'training_data'], inplace=True)

data_order =['expression.me_27k',
             'expression.me_450k',
             'me_27k.me_450k',
             'expression.me_27k.me_450k']

plu.plot_multi_omics_raw_results(results_df,
                                 axarr,
                                 data_order,
                                 metric=plot_metric)

handles = []
for ix, data in enumerate(data_order):
    handle = mpatches.Patch(color=sns.color_palette()[ix], label=data)
    handles.append(handle)

axarr[1, 2].legend(handles=handles, loc='lower right')


# ### Compare single-omics and multi-omics results

# In[14]:


# then, for each training data type, get the AUPR difference between signal and shuffled
compare_df = pd.DataFrame()
for training_data in results_df.training_data.unique():
    data_compare_df = au.compare_control_ind(
        results_df[results_df.training_data == training_data],
        identifier='identifier',
        metric=plot_metric,
        verbose=True
    )
    data_compare_df['training_data'] = training_data
    data_compare_df.rename(columns={'identifier': 'gene'}, inplace=True)
    compare_df = pd.concat((compare_df, data_compare_df))
    
compare_df.head(10)


# In[15]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (20, 12)})
sns.set_style('whitegrid')

# fig, axarr = plt.subplots(2, 4)
fig, axarr = plt.subplots(2, 3)
compare_df = (
    compare_df[compare_df.gene.isin(results_df.identifier.unique())]
    .sort_values(by=['gene', 'training_data'])
).copy()

data_names = {
    'expression': 'gene expression',
    'me_27k': '27K methylation',
    'me_450k': '450K methylation',
    'expression.me_27k': 'expression + 27K methylation',
    'expression.me_450k': 'expression + 450K methylation',
    'me_27k.me_450k': '27K methylation + 450K methylation',
    'expression.me_27k.me_450k': 'expression + 27K methylation + 450K methylation'
}

# we want to use the same colors as other figures for the individual
# data types, but different colors for the multi-omics models
colors = sns.color_palette()[:3] + sns.color_palette('Dark2')[:4]

plu.plot_multi_omics_results(compare_df,
                             axarr,
                             data_names,
                             colors,
                             metric=plot_metric)

handles = []
for ix, data in enumerate(list(data_names.values())):
    handle = mpatches.Patch(color=colors[ix], label=data)
    handles.append(handle)
    
plt.legend(title='Data types used to train model', handles=handles, loc='lower right')
plt.tight_layout()

if SAVE_FIGS:
    svg_filename = 'multi_omics_boxes_raw_feats.svg'
    png_filename = 'multi_omics_boxes_raw_feats.png'
    plt.savefig(images_dir / svg_filename, bbox_inches='tight')
    plt.savefig(images_dir / png_filename, dpi=300, bbox_inches='tight')


# ### Compare best-performing single-omics and multi-omics data types

# In[16]:


# for each data type, classify it as single-omics or multi-omics
compare_df['model_type'] = 'Best single-omics'
# multi-omics data types are concatenated using dots
compare_df.loc[compare_df.training_data.str.contains('\.'), 'model_type'] = 'Best multi-omics'
print(compare_df.training_data.unique())
compare_df[compare_df.gene == 'TP53'].head(10)


# In[17]:


sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')

plu.plot_best_multi_omics_results(compare_df,
                                  ylim=(0.0, 0.7),
                                  metric=plot_metric)

if SAVE_FIGS:
    svg_filename = 'multi_omics_best_model_raw_feats.svg'
    png_filename = 'multi_omics_best_model_raw_feats.png'
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / svg_filename, bbox_inches='tight')
    plt.savefig(images_dir / png_filename, dpi=300, bbox_inches='tight')

