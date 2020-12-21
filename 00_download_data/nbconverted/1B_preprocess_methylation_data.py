#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer methylation data

# Load the downloaded data and curate sample IDs.

# In[1]:


import os
import pandas as pd

import mpmp.config as cfg


# ### Load and process methylation data

# In[4]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head(1)


# In[5]:


tcga_methylation_df = pd.read_csv(
    os.path.join(cfg.raw_data_dir, manifest_df.loc['methylation'].filename),
    index_col=0, sep='\t')

print(tcga_methylation_df.shape)
tcga_methylation_df.iloc[:5, :5]


# In[6]:


# remove probes with missing values, and transpose to be a
# samples x probes matrix
tcga_methylation_df = (tcga_methylation_df
    .dropna(axis='rows')
    .transpose()
    .sort_index(axis='rows')
    .sort_index(axis='columns')
)

tcga_methylation_df.index.rename('sample_id', inplace=True)


# In[7]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_methylation_df.index = tcga_methylation_df.index.str.slice(start=0, stop=15)
tcga_methylation_df = tcga_methylation_df.loc[~tcga_methylation_df.index.duplicated(), :]


# In[8]:


print(tcga_methylation_df.shape)
tcga_methylation_df.iloc[:5, :5]


# In[9]:


tcga_methylation_df.to_csv(cfg.methylation_data, sep='\t', compression='gzip', float_format='%.3g')

