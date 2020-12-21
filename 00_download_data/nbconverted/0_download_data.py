#!/usr/bin/env python
# coding: utf-8

# ## Download multiple modalities of pan-cancer data from TCGA
# 
# The data is accessed directly from the [Genome Data Commons](https://gdc.cancer.gov/about-data/publications/pancanatlas).

# In[1]:


import os
import pandas as pd
from urllib.request import urlretrieve

import mpmp.config as cfg


# First, we load a manifest file containing the GDC API ID and filename for each relevant file, as well as the md5 checksum to make sure the whole/uncorrupted file was downloaded.
# 
# The manifest included in this GitHub repo was downloaded from https://gdc.cancer.gov/node/971 on December 1, 2020.

# In[2]:


manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head()


# ### Download gene expression data

# In[3]:


os.makedirs(cfg.raw_data_dir, exist_ok=True)
    
rnaseq_id, rnaseq_filename = manifest_df.loc['rna_seq'].id, manifest_df.loc['rna_seq'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(rnaseq_id)
exp_filepath = os.path.join(cfg.raw_data_dir, rnaseq_filename)

if not os.path.exists(exp_filepath):
    urlretrieve(url, exp_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[4]:


md5_sum = get_ipython().getoutput('md5sum $exp_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['rna_seq'].md5


# ### Download DNA methylation data

# In[5]:


me_id, me_filename = manifest_df.loc['methylation'].id, manifest_df.loc['methylation'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(me_id)
me_filepath = os.path.join(cfg.raw_data_dir, me_filename)

if not os.path.exists(me_filepath):
    urlretrieve(url, me_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[6]:


md5_sum = get_ipython().getoutput('md5sum $me_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['methylation'].md5


# ### Download RPPA data

# In[7]:


rppa_id, rppa_filename = manifest_df.loc['rppa'].id, manifest_df.loc['rppa'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(rppa_id)
rppa_filepath = os.path.join(cfg.raw_data_dir, rppa_filename)

if not os.path.exists(rppa_filepath):
    urlretrieve(url, rppa_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[8]:


md5_sum = get_ipython().getoutput('md5sum $rppa_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['rppa'].md5


# ### Download miRNA data

# In[9]:


mirna_id, mirna_filename = manifest_df.loc['mirna'].id, manifest_df.loc['mirna'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(mirna_id)
mirna_filepath = os.path.join(cfg.raw_data_dir, mirna_filename)

if not os.path.exists(mirna_filepath):
    urlretrieve(url, mirna_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[10]:


md5_sum = get_ipython().getoutput('md5sum $mirna_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['mirna'].md5
