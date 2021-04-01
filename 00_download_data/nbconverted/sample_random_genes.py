#!/usr/bin/env python
# coding: utf-8

# ## Random gene sampling

# In[1]:


import os

import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du


# In[2]:


sample_info_df = du.load_sample_info('expression', verbose=True)
mutation_df = du.load_pancancer_data(verbose=True)[1]
print(sample_info_df.shape)
print(mutation_df.shape)


# In[3]:


mutations_df = (mutation_df
    .merge(sample_info_df, how='inner', left_index=True, right_index=True)
    .drop(columns=['sample_type', 'id_for_stratification'])
)
print(mutations_df.shape)


# In[4]:


sum_df = mutations_df.groupby('cancer_type').agg('sum')
count_df = mutations_df.groupby('cancer_type').agg('count')
ratio_df = sum_df / count_df
sum_df.iloc[:5, :5]


# In[5]:


SUM_THRESHOLD = 10
PROP_THRESHOLD = 0.1

sum_df = (sum_df > SUM_THRESHOLD)
ratio_df = (ratio_df > PROP_THRESHOLD)
valid_df = sum_df & ratio_df

print(sum_df.sum().sum())
print(ratio_df.sum().sum())
valid_df.iloc[:5, :5]


# In[6]:


print(valid_df.sum().sum())
print(valid_df.sum().sort_values(ascending=False).head(10))


# In[7]:


NUM_CANCERS = 3

valid_genes = valid_df.sum()[valid_df.sum() >= NUM_CANCERS]
print(valid_genes.head(10))


# In[8]:


# sample randomly from valid genes and write to dataframe
sampled_genes = valid_genes.sample(n=50, random_state=cfg.default_seed)
print(sampled_genes.head())


# In[9]:


# get oncogene/TSG status from other gene lists
top50_df = du.load_top_50()
vogelstein_df = du.load_vogelstein()
gene_to_class_map = dict(zip(top50_df.gene, top50_df.classification))
for gene in vogelstein_df.gene:
    if gene not in gene_to_class_map:
        gene_to_class_map[gene] = vogelstein_df.loc[vogelstein_df.gene == gene, 'classification'].values[0]
        
print(list(gene_to_class_map.items())[:5])


# In[10]:


def get_class(gene):
    # if genes aren't in other gene lists, mark as 'neither'
    # we could do this in a more sophisticated way in the future, if we want
    try:
        return gene_to_class_map[gene]
    except KeyError:
        return 'neither'
    
classes = [get_class(gene) for gene in sampled_genes.index.values]
random_df = pd.DataFrame({
    'gene': sampled_genes.index.values,
    'classification': classes
}).set_index('gene')

random_df.head()


# In[11]:


random_df.to_csv(cfg.random_genes, sep='\t')

