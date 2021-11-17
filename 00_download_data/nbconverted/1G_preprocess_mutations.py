#!/usr/bin/env python
# coding: utf-8

# ## Preprocess TCGA mutation data
# 
# All this does is save the sample info for TCGA somatic mutation data, which is necessary to use mutation information as a predictor (e.g. in survival/prognosis prediction experiments).
# 
# This data was already preprocessed in http://github.com/greenelab/pancancer and is loaded directly from that repo, so no further preprocessing is necessary.

# In[1]:


import os

import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du
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


# ### Load and process somatic mutation data

# In[4]:


pancan_data = du.load_pancancer_data(verbose=True)
sample_freeze_df = pancan_data[0]

print(sample_freeze_df.duplicated(['SAMPLE_BARCODE']).sum())
assert (
    sample_freeze_df.duplicated(['SAMPLE_BARCODE']).sum() == 0
)

sample_freeze_df.set_index('SAMPLE_BARCODE', inplace=True)
sample_freeze_df.index.rename('sample_id', inplace=True)
sample_freeze_df.head()


# ### Process TCGA cancer type and sample type info from barcodes
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.

# In[5]:


# get sample info and save to file

tcga_id = tu.get_and_save_sample_info(sample_freeze_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict,
                                      training_data='mutation')

print(tcga_id.shape)
tcga_id.head()


# In[6]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_mutation_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()

