#!/usr/bin/env python
# coding: utf-8

# ## Preprocess TCGA mutational signatures
# 
# Load the downloaded data and curate sample IDs.
# 
# Mutational signature information for the TCGA whole-exome samples isn't available from GDC like the other datasets we're using, but we can get them from the [ICGC data portal here](https://dcc.icgc.org/releases/PCAWG/mutational_signatures/). These were originally generated in [this paper](https://www.nature.com/articles/s41586-020-1943-3).

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


# ### Load and process mutational signatures data

# In[4]:


# these are the "single base signatures" described in the paper linked above, 
# or for more information see: 
# https://cancer.sanger.ac.uk/cosmic/signatures/SBS/index.tt
# as far as I can tell, DBS and ID signatures weren't generated for TCGA whole-exome samples
url = (
    'https://dcc.icgc.org/api/v1/download'
    '?fn=/PCAWG/mutational_signatures/Signatures_in_Samples/SP_Signatures_in_Samples/'
    'TCGA_WES_sigProfiler_SBS_signatures_in_samples.csv'
)
mut_sigs_df = pd.read_csv(url, index_col=1)
mut_sigs_df.index.rename('sample_id', inplace=True)

print(mut_sigs_df.shape)
print(mut_sigs_df.columns)
mut_sigs_df.iloc[:5, :5]


# In[5]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
mut_sigs_df.index = mut_sigs_df.index.str.slice(start=0, stop=15)
mut_sigs_df = mut_sigs_df.loc[~mut_sigs_df.index.duplicated(), :]
print(mut_sigs_df.shape)


# In[6]:


(mut_sigs_df
    .drop(columns=['Cancer Types', 'Accuracy'])
    .to_csv(cfg.data_types['mut_sigs'], sep='\t')
)


# ### Process TCGA cancer type and sample type info from barcodes
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.

# In[7]:


# get sample info and save to file

tcga_id = tu.get_and_save_sample_info(mut_sigs_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict,
                                      training_data='mut_sigs')

print(tcga_id.shape)
tcga_id.head()


# In[8]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_mut_sigs_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()

