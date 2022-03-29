#!/usr/bin/env python
# coding: utf-8

# ## Explore TCGA normal tissue samples
# 
# We want to answer the following:
# 
# * How many healthy normals per cancer type
# * Do any of them have positive labels
# * Are these getting included in final dataset

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.utilities.data_utilities as du
import mpmp.utilities.tcga_utilities as tu


# In[2]:


# just do the analysis for expression data for now
# we could look at the overlap if we want to in the future
sample_info_df = du.load_sample_info('expression')
print(sample_info_df.sample_type.unique())
sample_info_df.head()


# In[3]:


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


# In[4]:


# load mutation data
pancancer_data = du.load_pancancer_data()
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data


# In[5]:


def get_mutation_df(gene, classification):
    include_copy = True
    if classification == "Oncogene":
        copy_number_df = copy_gain_df.loc[:, gene]
    elif classification == "TSG":
        copy_number_df = copy_loss_df.loc[:, gene]
    elif classification == "Oncogene, TSG":
        # some genes may act as both (i.e. in a cancer type-specific
        # or tissue-specific manner), in this case we'll just use the
        # union of the gain/loss dfs to define positive labeled samples
        copy_number_df = (
            copy_gain_df.loc[:, gene] | copy_loss_df.loc[:, gene]
        )
    else:
        copy_number_df = pd.DataFrame()
        include_copy = False
        
    y_df, _ = tu.process_y_matrix(
        mutation_df.loc[:, gene],
        copy_number_df,
        include_copy,
        gene,
        sample_freeze_df,
        mut_burden_df,
        filter_count=cfg.filter_count,
        filter_prop=cfg.filter_prop,
        output_directory=None,
        filter_cancer_types=False,
        hyper_filter=5,
    )
    return y_df
    
def check_normal_mutations(gene, classification='neither'):
    y_df = get_mutation_df(gene, classification)
    normal_samples = (sample_info_df
        [sample_info_df.sample_type.str.contains('Normal')].index.values
    )
    print(y_df.head())
    print(normal_samples[:5])
    print(len(normal_samples))
    y_df['is_normal'] = y_df.index.isin(normal_samples)
    print(y_df.is_normal.sum())
    normal_mut_df = (y_df[y_df.is_normal] 
      .groupby('DISEASE')
      .sum()
    )
    return normal_mut_df
    
df = check_normal_mutations('TP53', 'TSG')
df


# In[6]:


# check that normal samples are not in sample freeze
# this would make sense, and is how they are being excluded in our expts
# normal samples have sample type ID 10-19:
# https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/#tcga-barcode
(sample_freeze_df.SAMPLE_BARCODE.str.split('-', expand=True)[3].unique())


# So, to answer our questions:
# 
# * There are quite a few normal samples for which we have -omics data, the proportion varies by cancer type
# * None of them have mutation calling data, thus none of them have positive samples
# * These are not included in the sample freeze/mutation data so they are not being included in our experiments
