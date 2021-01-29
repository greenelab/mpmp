#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer 450K methylation data

# Load the downloaded data and curate sample IDs.
# 
# TODO: look at cancer type composition of represented samples (are any missing?)

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg


# ### Load and process methylation data

# In[2]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head(3)


# In[3]:


# this is much faster than loading directly from .tsv
# useful if you need to run this preprocessing script multiple times
methylation_pickle = os.path.join(cfg.data_dir, 'methylation_450k.pkl')

if os.path.isfile(methylation_pickle):
    print('loading from pickle')
    tcga_methylation_df = pd.read_pickle(methylation_pickle)
else:
    tcga_methylation_df = (
        pd.read_csv(os.path.join(cfg.raw_data_dir,
                                 manifest_df.loc['methylation_450k'].filename),
                    index_col=0,
                    sep='\t',
                    dtype='float32', # float64 won't fit in 64GB RAM
                    converters={0: str}) # don't convert the col names to float
           .transpose()
    )

tcga_methylation_df.index.rename('sample_id', inplace=True)

print(tcga_methylation_df.shape)
tcga_methylation_df.iloc[:5, :5]


# In[4]:


if os.path.isfile(methylation_pickle):
    print('pickle already exists')
else:
    print('saving df to pickle')
    tcga_methylation_df.to_pickle(methylation_pickle)


# In[5]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_methylation_df.index = tcga_methylation_df.index.str.slice(start=0, stop=15)
tcga_methylation_df = tcga_methylation_df.loc[~tcga_methylation_df.index.duplicated(), :]
print(tcga_methylation_df.shape)


# In[6]:


# how many missing values does each probe (column) have?
sample_na = tcga_methylation_df.transpose().isna().sum()
print(sample_na.shape)
sample_na.sort_values(ascending=False).head()


# In[7]:


# remove 10 samples with most NAs, then impute for probes with 1 or 2 NA values
n_filter = 10
n_impute = 5

samples_sorted = sample_na.sort_values(ascending=False)
output_dir = os.path.join(cfg.data_dir, 'methylation_preprocessed')
os.makedirs(output_dir, exist_ok=True)

def filter_na_samples(methylation_df, bad_samples):
    # don't drop NA columns, we'll do that after imputation
    return (
        methylation_df.loc[~methylation_df.index.isin(bad_samples)]
    )

def impute_leq(methylation_df, n_na):
    if n_na == 0:
        return methylation_df
    else:
        return methylation_df.fillna(methylation_df.mean(), limit=n_na)

# filter, impute, drop NA columns
print(tcga_methylation_df.shape)
samples_for_count = samples_sorted.iloc[:n_filter].index.values
tcga_methylation_df = filter_na_samples(tcga_methylation_df,
                                        samples_for_count)
print(tcga_methylation_df.shape)
tcga_methylation_df = impute_leq(tcga_methylation_df, n_impute)
tcga_methylation_df.dropna(axis='columns', inplace=True)
print(tcga_methylation_df.shape)


# In[9]:


from sklearn.decomposition import PCA

pca_dir = os.path.join(cfg.data_dir, 'me_compressed')
os.makedirs(pca_dir, exist_ok=True)

n_pcs_list = [100, 1000, 5000]

# it's much faster to just calculate this once for max n_pcs, and truncate it,
# than to recalculated it for each number of PCs we want
pca = PCA(n_components=max(n_pcs_list))
me_pca = pca.fit_transform(tcga_methylation_df)
print(me_pca.shape)

for n_pcs in n_pcs_list:
    me_pca_truncated = pd.DataFrame(me_pca[:, :n_pcs], index=tcga_methylation_df.index)
    print(me_pca_truncated.shape)
    me_pca_truncated.to_csv(
        os.path.join(pca_dir, 'me_450k_f{}_i{}_pc{}.tsv.gz'.format(
                         n_filter, n_impute, n_pcs)),
        sep='\t',
        float_format='%.3g')

