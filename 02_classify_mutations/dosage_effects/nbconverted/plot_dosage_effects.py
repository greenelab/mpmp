#!/usr/bin/env python
# coding: utf-8

# ## asdf

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
results_dir = Path(cfg.results_dirs['mutation'], 'dosage_effects')


# In[3]:


# process all component experiments
# control, drop_target, only_target
# TODO: explain
results_df = []
for subdir in results_dir.iterdir():
    experiment = subdir.stem
    gene_dir = Path(subdir, 'gene')
    model_results_df = au.load_stratified_prediction_results(gene_dir, 'gene')
    model_results_df['experiment'] = experiment
    results_df.append(model_results_df)
    
results_df = pd.concat(results_df)
print(results_df.shape)
results_df.head()


# In[4]:


# try plot with no normalization/correction for shuffled baseline
sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')

plot_df = results_df[((results_df.signal == 'signal') &
                      (results_df.data_type == 'test'))]
hue_order = [
    'control',
    'drop_target',
    'only_target'
]
sns.boxplot(data=plot_df, x='identifier', y='aupr', hue='experiment',
            hue_order=hue_order)


# In[5]:


# now correct for shuffled baseline
results_df = (results_df
    .drop(columns='training_data')
    .rename(columns={'experiment': 'training_data'})
)
all_results_df = au.compare_all_data_types(results_df,
                                           0.05,
                                           filter_genes=False,
                                           compare_ind=True)
all_results_df = all_results_df.rename(columns={'training_data': 'experiment'})
all_results_df.head()


# In[6]:


sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')

hue_order = [
    'control',
    'drop_target',
    'only_target'
]
sns.boxplot(data=all_results_df, x='gene', y='delta_aupr', hue='experiment',
            hue_order=hue_order)


# In[7]:


pivot_df = all_results_df.pivot(index=['gene', 'seed', 'fold'],
                                columns='experiment',
                                values='delta_aupr')
pivot_df.head()


# In[8]:


pivot_df['drop_target'] = pivot_df.control - pivot_df.drop_target
pivot_df['only_target'] = pivot_df.control - pivot_df.only_target

pivot_df = (pivot_df
  .drop(columns='control')
  .reset_index()
  .melt(id_vars=['gene', 'seed', 'fold'],
        value_vars=['drop_target', 'only_target'],
        value_name='delta_aupr')
)

pivot_df.head()


# In[9]:


sns.set({'figure.figsize': (13, 6)})
sns.set_style('whitegrid')
sns.set_palette('Set2')

hue_order = [
    'drop_target',
    'only_target'
]
sns.barplot(data=pivot_df, x='gene', y='delta_aupr', hue='experiment',
            hue_order=hue_order)

