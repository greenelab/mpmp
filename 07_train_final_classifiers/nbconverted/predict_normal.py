#!/usr/bin/env python
# coding: utf-8

# ## Use pre-trained models to make predictions on normal tissue samples
# 
# For some cancer types, TCGA provides samples from normal tissue in addition to the tumor samples (see `01_explore_data/normal_tissue_samples.ipynb`).
# 
# In this analysis, we want to make predictions on those samples and compare them to our tumor sample predictions.
# 
# Our assumption is that our models will predict that the normal tissue samples have a low probability of mutation (since they almost certainly do not have somatic mutations in any of the genes of interest).

# In[6]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.data_utilities as du
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[7]:


results_dir = Path(cfg.results_dirs['final'],
                   'pilot_genes',
                   'gene').resolve()

genes = [g.stem for g in results_dir.iterdir() if not g.is_file()]
print(genes)


# In[8]:


# load expression sample info, this has tumor/normal labels
sample_info_df = du.load_sample_info('expression')
print(sample_info_df.sample_type.unique())
sample_info_df.head()


# In[9]:


# load expression data
data_df = du.load_raw_data('expression', verbose=True)
print(data_df.shape)
data_df.iloc[:5, :5]


# In[11]:


normal_ids = (
    sample_info_df[sample_info_df.sample_type.str.contains('Normal')]
      .index
      .intersection(data_df.index)
)
print(len(normal_ids))
print(normal_ids[:5])

