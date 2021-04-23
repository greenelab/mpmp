#!/usr/bin/env python
# coding: utf-8

# ## Preprocess miRNA data

# Load the downloaded data, update gene identifiers to Entrez, and curate sample IDs.

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


# ### Load and process miRNA data

# In[4]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)

# we have sample info for the mirna data (mirna_sample), and the data itself (mirna)
manifest_df.filter(like='mirna', axis=0)


# In[5]:


tcga_mirna_df = pd.read_csv(
    os.path.join(cfg.raw_data_dir, manifest_df.loc['mirna'].filename),
    index_col=0, sep=',')

print(tcga_mirna_df.shape)
tcga_mirna_df.head()


# ### Process gene expression matrix
# 
# This involves processing sample IDs, sorting and subsetting.

# In[6]:


# remove transcripts with NA values
tcga_mirna_df = (tcga_mirna_df
    .drop(columns=['Correction'])
    .dropna(axis='rows')
    .groupby(level=0).mean()
    .transpose()
    .sort_index(axis='rows')
    .sort_index(axis='columns')
)

tcga_mirna_df.index.rename('sample_id', inplace=True)


# In[7]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_mirna_df.index = tcga_mirna_df.index.str.slice(start=0, stop=15)
tcga_mirna_df = tcga_mirna_df.loc[~tcga_mirna_df.index.duplicated(), :]


# In[8]:


print(tcga_mirna_df.shape)
tcga_mirna_df.head()


# ### Process TCGA cancer type and sample type info from barcodes
# 
# Cancer-type includes `OV`, `BRCA`, `LUSC`, `LUAD`, etc. while sample-type includes `Primary`, `Metastatic`, `Solid Tissue Normal`, etc.
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.
# 
# The goal is to use this info to stratify train and test sets by cancer type and sample type. 

# In[9]:


# get sample info and save to file
tcga_id = tu.get_and_save_sample_info(tcga_mirna_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict,
                                      training_data='mirna')

print(tcga_id.shape)
tcga_id.head()


# In[10]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_mirna_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()


# In[11]:


mirna_file = os.path.join(cfg.data_dir, 'tcga_mirna_matrix_processed.tsv')
tcga_mirna_df.to_csv(mirna_file, sep='\t', float_format='%.3g')

