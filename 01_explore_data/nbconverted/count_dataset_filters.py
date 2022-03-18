#!/usr/bin/env python
# coding: utf-8

# ## Count number of valid genes after applying cancer type filters
# 
# In [the preprocessing code for our classifiers](https://github.com/greenelab/mpmp/blob/5d5fa0823b00fc3080d3a9db69d8d6704f554549/mpmp/utilities/tcga_utilities.py#L84), we originally filtered out cancer types that don't contain at least 5% of samples mutated and at least 10 total samples mutated, for a given target gene.
# 
# Here, we want to try a different approach (TODO: describe)

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# load sample info and mutation data, this takes some time
tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=False)
pancancer_data = du.load_pancancer_data()


# In[3]:


# load merged gene set info
genes_df = du.load_merged()
genes_df.head()


# In[7]:


def gene_mutation_count(gene,
                        data_model,
                        classification='neither',
                        filter_cancer_types=False):
    """Count mutations for a given gene."""
    try:
        tcga_data.process_data_for_gene(
            gene,
            classification,
            './gene_mutation_counts',
            filter_cancer_types=filter_cancer_types
        )
        count_df = pd.read_csv(
            './gene_mutation_counts/{}_filtered.tsv'.format(gene),
            sep='\t'
        )
        return count_df.loc[0].values.tolist()
    except KeyError:
        return (0, 0, False)
        
gene_mutation_count('TP53', tcga_data, 'TSG')


# In[14]:


gene_counts_df = []

for g_id, g_series in genes_df.iterrows():
    gene_info = gene_mutation_count(
        g_series.gene,
        tcga_data,
        g_series.classification
    )
    gene_counts_df.append(gene_info)
    
gene_counts_df = pd.DataFrame(
    gene_counts_df,
    columns=['count', 'prop', 'filter_gene']
)
print(gene_counts_df.shape)
gene_counts_df.head()

