#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer RPPA data
# 
# Load the downloaded data and curate sample IDs.

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.tcga_utilities as tu


# ### Read TCGA Barcode Curation Information
# 
# Extract information from TCGA barcodes - `cancer-type` and `sample-type`. See https://github.com/cognoma/cancer-data for more details

# In[2]:


(cancer_types_df,
 cancertype_codes_dict,
 sample_types_df,
 sampletype_codes_dict) = tu.get_tcga_barcode_info()
cancer_types_df.head(2)


# In[3]:


sample_types_df.head(2)


# ### Load and process RPPA data

# In[4]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head(5)


# In[5]:


tcga_rppa_df = (
    pd.read_csv(os.path.join(cfg.raw_data_dir, manifest_df.loc['rppa'].filename),
                index_col=0, sep='\t')
)

tcga_rppa_df.index.rename('sample_id', inplace=True)

print(tcga_rppa_df.shape)
tcga_rppa_df.iloc[:5, :5]


# In[6]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_rppa_df.index = tcga_rppa_df.index.str.slice(start=0, stop=15)
tcga_rppa_df = tcga_rppa_df.loc[~tcga_rppa_df.index.duplicated(), :]
print(tcga_rppa_df.shape)


# ### Filtering/imputation/NA removal

# In[7]:


# how many missing values does each probe (column) have?
probe_na = tcga_rppa_df.isna().sum()
print(probe_na.shape)
print(probe_na.sort_values(ascending=False).head(10))
sns.set()
sns.histplot(probe_na)


# In[8]:


# how many missing values does each sample have?
sample_na = tcga_rppa_df.transpose().isna().sum()
print(sample_na.shape)
print(sample_na.sort_values(ascending=False).head(10))
sns.set()
sns.histplot(sample_na)


# In[9]:


# for now, just drop all columns with NAs, since there's only a few
print(tcga_rppa_df.shape)
tcga_rppa_df.dropna(axis='columns', inplace=True)
print(tcga_rppa_df.shape)

# check for duplicate samples
print('Duplicates:', np.count_nonzero(tcga_rppa_df.index.duplicated()))


# ### Process TCGA cancer type and sample type info from barcodes
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.

# In[10]:


# get sample info and save to file

tcga_id = tu.get_and_save_sample_info(tcga_rppa_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict,
                                      training_data='rppa')

print(tcga_id.shape)
tcga_id.head()


# In[11]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_rppa_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()


# In[12]:


# compare cancer types from sample ID to cancer types provided with data matrix
type_df = (tcga_rppa_df
    .merge(tcga_id, left_index=True, right_on='sample_id')
    .set_index('sample_id')
    .drop(columns=['sample_type', 'id_for_stratification'])
)
neq_df = type_df[type_df.TumorType != type_df.cancer_type]
print(neq_df.shape)
print(neq_df.TumorType.unique())
print(neq_df.cancer_type.unique())
neq_df.head()


# Looks like "CORE" is short for "COAD/READ", so all the cancer types match. So, we can just drop the TumorType column and use the sample info saved in the data directory.

# In[13]:


rppa_file = os.path.join(cfg.data_dir, 'tcga_rppa_matrix_processed.tsv')
(tcga_rppa_df
    .drop(columns=['TumorType'])
    .to_csv(rppa_file, sep='\t', float_format='%.3g')
)

