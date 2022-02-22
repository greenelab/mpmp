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


tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=False)


# In[3]:


def gene_sample_count(gene, data_model, classification='neither'):
    try:
        tcga_data.process_data_for_gene(gene,
                                        classification,
                                        None)
        sample_count = tcga_data.X_df.shape[0]
    except KeyError:
        sample_count = np.nan
        
    # TODO: get cancer types?
    return (gene, sample_count)


# In[4]:


# cache partial results and load them
output_file = Path('./gene_sample_count.tsv')
if output_file.is_file():
    output_df = pd.read_csv(output_file, sep='\t', index_col=0)
else:
    output_df = pd.DataFrame()
    
print(output_df.shape)
output_df.head()


# In[5]:


print(gene_sample_count('TP53', tcga_data, classification='TSG'))


# In[6]:


gene_df = du.load_merged()
gene_df.head()


# In[7]:


save_every = 50

for gene_ix, gene_series in gene_df.iterrows():
    
    # if gene has already been processed, skip it
    if gene_series.gene in output_df.index:
        continue
        
    # load sample count for gene
    gene, sample_count = gene_sample_count(
        gene_series.gene,
        tcga_data,
        gene_series.classification)
    
    # add to output dataframe
    output_df = pd.concat((
        output_df,
        pd.DataFrame(sample_count,
                     index=[gene],
                     columns=['sample_count'])
    ))
    
    # save results every save_every genes, and at the end of all genes
    # this allows us to restart if this runs for a while and gets interrupted
    progress_ix = gene_ix + 1
    if ((progress_ix % save_every == 0) or (progress_ix == gene_df.shape[0])) and (gene_ix != 0):
        print('processed: {} / {}'.format(gene_ix+1, gene_df.shape[0]),
              file=sys.stderr)
        output_df.to_csv(output_file, sep='\t')


# In[8]:


print(output_df.shape)
output_df.head()

