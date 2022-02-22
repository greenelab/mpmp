#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.utilities.data_utilities as du


# In[2]:


tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=True)


# In[8]:


def gene_sample_count(gene, data_model, classification='neither'):
    print(gene, file=sys.stderr)
    try:
        tcga_data.process_data_for_gene(gene,
                                        classification,
                                        None)
        sample_count = tcga_data.X_df.shape[0]
    except KeyError:
        sample_count = np.nan
        
    # TODO: get cancer types?
    return (gene, sample_count)


# In[5]:


# cache partial results and load them
output_file = Path('./gene_sample_count.tsv')
if output_file.is_file():
    output_df = pd.read_csv(output_file, sep='\t', index_col=0)
else:
    output_df = pd.DataFrame()
    
print(output_df.shape)
output_df.head()


# In[9]:


print(gene_sample_count('TP53', tcga_data, classification='TSG'))


# In[6]:


vogelstein_df = du.load_vogelstein()
vogelstein_df.head()


# In[ ]:


save_every = 50

for gene_ix, gene_series in vogelstein_df.iterrows():
    if (gene_ix % save_every == 0) and (gene_ix != 0):
        output_df.to_csv(output_file, sep='\t')
    
        

