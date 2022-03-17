#!/usr/bin/env python
# coding: utf-8

# ## Explore TCGA healthy control samples
# 
# We want to answer the following:
# 
# * how many healthy controls per cancer type
# * do any of them have positive labels
# * are these getting included in final dataset (probably?)

# In[2]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.utilities.data_utilities as du

# load sample info and mutation data, this takes some time
tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=False)
pancancer_data = du.load_pancancer_data()
# In[7]:


# just do the analysis for expression data for now
# we could look at the overlap if we want to in the future
sample_info_df = du.load_sample_info('me_27k')
print(sample_info_df.sample_type.unique())
sample_info_df.head()


# In[16]:


# all of the normal samples have the term "Normal" in their sample_type
# see: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
normal_count_df = (sample_info_df
    .assign(normal_count=sample_info_df.sample_type.str.contains('Normal'))
    .groupby('cancer_type')
    .sum()
    # .drop(columns='id_for_stratification')
    # .rename(columns={'sample_type': 'normal_count'})
)

cancer_count_df = (sample_info_df
    .assign(cancer_count=(~sample_info_df.sample_type.str.contains('Normal')))
    .groupby('cancer_type')
    .sum()
)

count_df = normal_count_df.merge(cancer_count_df, left_index=True, right_index=True)
count_df['normal_prop'] = (
    count_df.normal_count / (count_df.cancer_count + count_df.normal_count)
)
count_df

