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


# ### Choose whether or not to use CNV info
# 
# Normally we'd use copy gain for oncogenes and copy loss for TSGs, but since we don't have these annotations for all the genes in the genome, there are two ways to handle marking samples as "mutated":
# 
# 1. Just use point mutations for all genes, no copy number info
# 2. Use point mutations and copy gain/loss for all genes
# 
# We provide an option here to run either of these.

# In[2]:


use_copy_data = False

if use_copy_data:
    output_file = Path('./gene_sample_count_with_copy.tsv')
else:
    output_file = Path('./gene_sample_count_no_copy.tsv')


# In[3]:


# load sample info and mutation data, this takes some time
tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=False)
pancancer_data = du.load_pancancer_data()


# In[4]:


def gene_sample_count(gene, data_model, classification='neither'):
    """Count valid samples/cancer types for a given gene."""
    try:
        tcga_data.process_data_for_gene(gene,
                                        classification,
                                        None)
        sample_count = tcga_data.X_df.shape[0]
        cancer_types = ';'.join(tcga_data.sample_info_df
          .loc[tcga_data.X_df.index, 'cancer_type']
          .unique()
        )
    except KeyError:
        sample_count = np.nan
        cancer_types = pd.NA
        
    return (gene, sample_count, cancer_types)


# In[5]:


# cache partial results and load them
# we're running this script for 20,000 genes, so it's nice to save progress in case
# execution is interrupted
if output_file.is_file():
    output_df = pd.read_csv(
        output_file, sep='\t', index_col=0
    )
    output_df.cancer_types.fillna('', inplace=True)
    output_df.loc[output_df.cancer_types == '[]', 'cancer_types'] = []
else:
    output_df = pd.DataFrame()
    
print(output_df.shape)
output_df.head()


# In[6]:


print(gene_sample_count('TP53', tcga_data, classification='TSG'))


# ### Calculate sample/cancer type count for all genes in the mutation gene set

# In[7]:


mutation_df = pancancer_data[1]
mutation_df.iloc[:5, :5]


# In[8]:


save_every = 100

gene_list = mutation_df.columns
for gene_ix, gene in enumerate(gene_list):
    
    # if gene has already been processed, skip it
    if gene in output_df.index:
        continue
        
    # load sample count for gene
    gene_name, sample_count, cancer_types = gene_sample_count(
        gene,
        tcga_data,
        classification=('Oncogene, TSG' if use_copy_data else 'neither')
    )
    
    # add to output dataframe
    output_df = pd.concat((
        output_df,
        pd.DataFrame([[sample_count, cancer_types]],
                     index=[gene_name],
                     columns=['sample_count', 'cancer_types'])
    ))
    
    # save results every save_every genes, and at the end of all genes
    # this allows us to restart if this runs for a while and gets interrupted
    progress_ix = gene_ix + 1
    if ((progress_ix % save_every == 0) or (progress_ix == len(gene_list))) and (gene_ix != 0):
        print('processed: {} / {}'.format(gene_ix+1, len(gene_list)),
              file=sys.stderr)
        output_df.to_csv(output_file, sep='\t')


# In[9]:


print(output_df.shape)
output_df.head()


# In[10]:


valid_genes = (output_df.sample_count > 0).sum()
print('Valid genes:', valid_genes, '/', output_df.shape[0])


# ### Plot distribution of valid cancer types across gene set

# In[11]:


output_df['num_cancer_types'] = output_df.cancer_types.str.split(';', expand=False).apply(len)
output_df.loc[output_df.sample_count == 0, 'num_cancer_types'] = 0
output_df.head()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set({'figure.figsize': (8, 6)})

sns.histplot(output_df.num_cancer_types, binwidth=1)
plt.xticks(range(0, output_df.num_cancer_types.max()+1))
plt.setp(plt.gca().get_xticklabels()[1::2], visible=False)
plt.xlabel('Cancer types')
plt.title('Number of valid cancer types, per gene')


# In[13]:


sns.set({'figure.figsize': (8, 6)})

sns.histplot(output_df.num_cancer_types, binwidth=1)
plt.xticks(range(0, output_df.num_cancer_types.max()+1))
plt.setp(plt.gca().get_xticklabels()[1::2], visible=False)
plt.xlabel('Cancer types')
plt.yscale('log')
plt.title('Number of valid cancer types, per gene, log scale')

