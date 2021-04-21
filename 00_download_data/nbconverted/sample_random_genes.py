#!/usr/bin/env python
# coding: utf-8

# ## Gene sampling for expression data comparisons
# 
# For mutation prediction from gene expression data, we want to compare the Vogelstein et al. 2013 gene set [located here](https://github.com/greenelab/pancancer/blob/master/data/vogelstein_cancergenes.tsv) with a set of random genes of equal length, and with a set of the most frequently mutated genes in TCGA of equal length.
# 
# This script will sample the random genes and identify the most frequently mutated genes, and generate files containing those genes and their classification information (oncogene/TSG/neither).

# In[1]:


import os

import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du


# In[2]:


# this is the number of valid genes in the Vogelstein gene set
NUM_GENES = 107

# sample random genes from set of genes with every gene with >= NUM_CANCERS
# valid cancer types
#
# if we sampled them randomly from all genes, it's likely that many of them
# would end up with no valid cancer types (i.e. not enough mutations to train
# a classifier), so we add this criterion to make sure they're reasonably
# frequently mutated
NUM_CANCERS = 2


# ### Load mutation and sample/cancer type info

# In[3]:


sample_info_df = du.load_sample_info('expression', verbose=True)
mutation_df = du.load_pancancer_data(verbose=True)[1]
print(sample_info_df.shape)
print(mutation_df.shape)


# In[4]:


mutations_df = (mutation_df
    .merge(sample_info_df, how='inner', left_index=True, right_index=True)
    .drop(columns=['sample_type', 'id_for_stratification'])
)
print(mutations_df.shape)


# ### Get number of mutations per gene, per cancer type

# In[5]:


sum_df = mutations_df.groupby('cancer_type').agg('sum')
count_df = mutations_df.groupby('cancer_type').agg('count')
ratio_df = sum_df / count_df
sum_df.iloc[:5, :5]


# In[6]:


SUM_THRESHOLD = 10
PROP_THRESHOLD = 0.1

sum_df = (sum_df > SUM_THRESHOLD)
ratio_df = (ratio_df > PROP_THRESHOLD)
valid_df = sum_df & ratio_df

print(sum_df.sum().sum())
print(ratio_df.sum().sum())
valid_df.iloc[:5, :5]


# ### Sample randomly from set of all valid genes

# In[8]:


valid_genes = valid_df.sum()[valid_df.sum() >= NUM_CANCERS]
print(valid_genes.head(10))
print(len(valid_genes))


# In[9]:


# sample randomly from valid genes and write to dataframe
sampled_genes = valid_genes.sample(n=NUM_GENES, random_state=cfg.default_seed)
print(sampled_genes.head())


# In[11]:


# get oncogene/TSG status from Vogelstein gene list
# this is just used to decide whether to add copy number gains/losses in mutation labeling
vogelstein_df = du.load_vogelstein()
gene_to_class_map = dict(zip(vogelstein_df.gene, vogelstein_df.classification))

def get_class(gene):
    # if genes aren't in other gene lists, mark as 'neither'
    try:
        return gene_to_class_map[gene]
    except KeyError:
        return 'neither'
    
random_classes = [get_class(gene) for gene in sampled_genes.index.values]

random_df = pd.DataFrame({
    'gene': sampled_genes.index.values,
    'classification': random_classes
}).set_index('gene')

random_df.head()


# In[12]:


random_df.to_csv(cfg.random_genes, sep='\t')


# ### Get top mutated genes
# 
# Same methods as in https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/top-50-pancanatlas-mutations.ipynb (but we want more than 50 genes, since we want a gene set of the same size as Vogelstein)

# In[13]:


mutation_count_df = mutation_df.sum().sort_values(ascending=False)
mutation_count_df.head()


# In[14]:


top_genes = mutation_count_df[:NUM_GENES]
top_classes = [get_class(gene) for gene in top_genes.index.values]
top_df = pd.DataFrame({
    'gene': top_genes.index.values,
    'classification': top_classes
}).set_index('gene')
top_df.head()


# In[15]:


top_df.to_csv(cfg.top_genes, sep='\t')

