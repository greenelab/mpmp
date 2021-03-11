#!/usr/bin/env python
# coding: utf-8

# ## Explore tumor purity data

# In[5]:


import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du


# In[6]:


purity_df = pd.read_csv(cfg.tumor_purity_data, sep='\t', index_col=0)
purity_df.head()


# In[7]:


# visualize distribution of tumor purity values
sns.set({'figure.figsize': (8, 5)})
sns.histplot(purity_df, x='purity', element='step')
plt.gca().axvline(x=purity_df.purity.median(), linestyle='--', color='red')
plt.xlabel('Tumor purity')


# In[8]:


# visualize distribution of tumor purity values, by cancer type
sample_info_df = du.load_sample_info('expression')
sample_info_df.head()


# In[11]:


purity_cancer_df = (purity_df[['purity']]
    .merge(sample_info_df, left_index=True, right_index=True)
    .drop(columns=['id_for_stratification'])
)
purity_cancer_df.head()


# In[45]:


sns.set({'figure.figsize': (8, 5)})

g = sns.kdeplot(data=purity_cancer_df, x='purity', hue='cancer_type', legend=False)
# TODO: figure out legend
# g.fig.legend(labels=[], ncol=2)
plt.gca().axvline(x=purity_df.purity.median(), linestyle='--', color='red')
plt.xlabel('Tumor purity')

