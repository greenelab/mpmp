#!/usr/bin/env python
# coding: utf-8

# ## Count number of valid genes after applying cancer type filters
# 
# In [the preprocessing code for our classifiers](https://github.com/greenelab/mpmp/blob/5d5fa0823b00fc3080d3a9db69d8d6704f554549/mpmp/utilities/tcga_utilities.py#L84), we filter out cancer types that don't contain at least 5% of samples mutated and at least 10 total samples mutated, for a given target gene.
# 
# We were curious how many total genes these filters would give us, if we look at _all_ ~20,000 genes we have mutation data for. This script filters samples for each gene and counts the number of samples/cancer types that would be included in our classifiers.

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


# In[4]:


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
    except KeyError:
        return
        
gene_mutation_count('TP53', tcga_data, 'TSG')

