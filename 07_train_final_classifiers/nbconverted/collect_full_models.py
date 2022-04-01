#!/usr/bin/env python
# coding: utf-8

# ## Collect full models and save into tsv files
# 
# As a resource to provide with our paper, we want to provide the fit models (coefficients/effect sizes, and parameter choices) as a data file.
# 
# This notebook collects the results of the bash script at `07_train_final_classifiers/scripts/run_all_genes.sh` and assembles them into dataframes/`.tsv` files.

# In[3]:


from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


results_dir = Path(cfg.results_dirs['final'],
                   'merged_genes',
                   'gene').resolve()


# ### Load coefficients and assemble into dataframe

# In[14]:


coefs = {}

for gene_dir in results_dir.iterdir():
    gene_name = gene_dir.stem
    gene_dir = Path(results_dir, gene_dir)
    if gene_dir.is_file(): continue
    for results_file in gene_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if 'coefficients' not in results_filename: continue
        coefs_df = pd.read_csv(results_file, sep='\t')
        coefs[gene_name] = coefs_df.loc[:, ['feature', 'weight']]
        
print(list(coefs.keys())[:5])
print(len(coefs.keys()))
coefs['TP53'].head()

