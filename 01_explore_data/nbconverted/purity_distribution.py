#!/usr/bin/env python
# coding: utf-8

# ## Explore tumor purity data

# In[1]:


import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du


# In[2]:


purity_df = pd.read_csv(cfg.tumor_purity_data, sep='\t', index_col=0)
purity_df.head()


# In[3]:


# visualize distribution of tumor purity values
sns.set({'figure.figsize': (8, 5)})
sns.histplot(purity_df, x='purity', element='step')
plt.gca().axvline(x=purity_df.purity.median(), linestyle='--', color='red')
plt.xlabel('Tumor purity')


# The median tumor purity value seems to be a bit higher than 0.5 (this is probably due to filtering in some cancer types for high purity samples).
# 
# This should be reasonable enough to start with for classification, although I don't think it's the best way to do predictive modeling of this data. Really, we should just do regression, but we could also explore other binarization methods, or binarizing by cancer type.

# In[4]:


# visualize distribution of tumor purity values, by cancer type
sample_info_df = du.load_sample_info('expression')
sample_info_df.head()


# In[5]:


purity_cancer_df = (purity_df[['purity']]
    .merge(sample_info_df, left_index=True, right_index=True)
    .drop(columns=['id_for_stratification'])
)
purity_cancer_df.head()


# In[6]:


sns.set({'figure.figsize': (8, 5)})

g = sns.kdeplot(data=purity_cancer_df, x='purity', hue='cancer_type', legend=False)
# TODO: figure out legend
# g.fig.legend(labels=[], ncol=2)
plt.gca().axvline(x=purity_df.purity.median(), linestyle='--', color='red')
plt.xlabel('Tumor purity')


# When we facet by cancer type, we can see that quite a few cancers have a peak closer to 0.8/0.9, probably due to filtering for high-purity samples. Others have many more low-purity samples (e.g. [pancreatic cancer](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/studied-cancers/pancreatic)), and some are roughly centered around 0.5.
# 
# In the future, we could look further into these differences between cancer types, depending on the results of the prediction experiments.
