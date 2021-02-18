#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer 450K methylation data

# Load the downloaded data and curate sample IDs.

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.tcga_utilities as tu


# ### Read TCGA Barcode Curation Information
# 
# Extract information from TCGA barcodes - `cancer-type` and `sample-type`. See https://github.com/cognoma/cancer-data for more details

# In[2]:


(cancer_types_df,
 cancertype_codes_dict,
 sample_types_df,
 sampletype_codes_dict) = tu.get_tcga_barcode_info()
cancer_types_df.head(2)


# In[3]:


sample_types_df.head(2)


# ### Load and process methylation data

# In[4]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head(3)


# In[5]:


# this is much faster than loading directly from .tsv
# useful if you need to run this preprocessing script multiple times
methylation_pickle = os.path.join(cfg.data_dir, 'methylation_450k.pkl')

if os.path.isfile(methylation_pickle):
    print('loading from pickle')
    tcga_methylation_df = pd.read_pickle(methylation_pickle)
else:
    tcga_methylation_df = (
        pd.read_csv(os.path.join(cfg.raw_data_dir,
                                 manifest_df.loc['methylation_450k'].filename),
                    index_col=0,
                    sep='\t',
                    dtype='float32', # float64 won't fit in 64GB RAM
                    converters={0: str}) # don't convert the col names to float
           .transpose()
    )

tcga_methylation_df.index.rename('sample_id', inplace=True)

print(tcga_methylation_df.shape)
tcga_methylation_df.iloc[:5, :5]


# In[6]:


if os.path.isfile(methylation_pickle):
    print('pickle already exists')
else:
    print('saving df to pickle')
    tcga_methylation_df.to_pickle(methylation_pickle)


# In[7]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_methylation_df.index = tcga_methylation_df.index.str.slice(start=0, stop=15)
tcga_methylation_df = tcga_methylation_df.loc[~tcga_methylation_df.index.duplicated(), :]
print(tcga_methylation_df.shape)


# In[8]:


# how many missing values does each sample have?
sample_na = tcga_methylation_df.transpose().isna().sum()
print(sample_na.shape)
sample_na.sort_values(ascending=False).head()


# In[9]:


# remove 10 samples with most NAs, then impute for probes with 1 or 2 NA values
n_filter = 10
n_impute = 5

samples_sorted = sample_na.sort_values(ascending=False)
output_dir = os.path.join(cfg.data_dir, 'methylation_preprocessed')
os.makedirs(output_dir, exist_ok=True)

def filter_na_samples(methylation_df, bad_samples):
    # don't drop NA columns, we'll do that after imputation
    return (
        methylation_df.loc[~methylation_df.index.isin(bad_samples)]
    )

def impute_leq(methylation_df, n_na):
    if n_na == 0:
        return methylation_df
    else:
        return methylation_df.fillna(methylation_df.mean(), limit=n_na)

# filter, impute, drop NA columns
print(tcga_methylation_df.shape)
samples_for_count = samples_sorted.iloc[:n_filter].index.values
tcga_methylation_df = filter_na_samples(tcga_methylation_df,
                                        samples_for_count)
print(tcga_methylation_df.shape)
tcga_methylation_df = impute_leq(tcga_methylation_df, n_impute)
tcga_methylation_df.dropna(axis='columns', inplace=True)
print(tcga_methylation_df.shape)


# ### Process TCGA cancer type and sample type info from barcodes
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.

# In[10]:


# get sample info and save to file
tcga_id = tu.get_and_save_sample_info(tcga_methylation_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict,
                                      training_data='me_450k')

print(tcga_id.shape)
tcga_id.head()


# In[11]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_me_450k_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()


# ### Dimension "trimming" by MAD
# 
# Like with gene expression data where we generally use the top 8,000 (or so) genes by mean absolute deviation, one way to cut down on some of the dimensionality of this dataset is to filter features by MAD. Here, we'll filter to the top 100K features.

# In[12]:


mad_genes = tcga_methylation_df.mad(axis=0)
mad_genes.sort_values(ascending=False, inplace=True)
print(mad_genes.iloc[:10])


# In[13]:


n_mad_genes = 100000
me_mad_df = tcga_methylation_df.loc[:, mad_genes.iloc[:n_mad_genes].index]
me_mad_df.to_pickle(os.path.join(cfg.data_dir, 
                                 'methylation_450k_f{}_i{}_mad{}.pkl'.format(
                                     n_filter, n_impute, n_mad_genes)))


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.histplot(data=mad_genes)
plt.xlabel('Mean absolute deviation')
plt.ylabel('Count')


# ### Dimension reduction
# 
# Since we can't realistically fit a logistic regression model to 300K features, compress the data using PCA with various dimensions, and save the results to tsv files.
# 
# Note that it's much faster to calculate this once for the largest dimensionality, then truncate it, than to recalculate the PCA for each number of PCs. However, using the sklearn PCA implementation, the results are actually slightly different for these two approaches (particularly for smaller PCs), since it uses a randomized thin SVD algorithm by default rather than calculating the exact SVD. 
# 
# I created [this issue](https://github.com/greenelab/mpmp/issues/15) to investigate/remind myself of this instability in the future, but I don't think it'll matter that much in practice.

# In[15]:


from sklearn.decomposition import PCA

pca_dir = os.path.join(cfg.data_dir, 'me_compressed')
os.makedirs(pca_dir, exist_ok=True)

n_pcs_list = [100, 1000, 5000]

# it's much faster to just calculate this once for max n_pcs, and truncate it,
# than to recalculate it for each number of PCs we want
pca = PCA(n_components=max(n_pcs_list), random_state=cfg.default_seed)
me_pca = pca.fit_transform(tcga_methylation_df)
print(me_pca.shape)

for n_pcs in n_pcs_list:
    me_pca_truncated = pd.DataFrame(me_pca[:, :n_pcs], index=tcga_methylation_df.index)
    print(me_pca_truncated.shape)
    me_pca_truncated.to_csv(
        os.path.join(pca_dir, 'me_450k_f{}_i{}_pc{}.tsv.gz'.format(
                         n_filter, n_impute, n_pcs)),
        sep='\t',
        float_format='%.3g')


# In[16]:


# plot PCA variance explained
import matplotlib.pyplot as plt
import seaborn as sns

sns.set({'figure.figsize': (15, 4)})
fig, axarr = plt.subplots(1, 3)

for ix, n_pcs in enumerate(n_pcs_list):
    ve = pca.explained_variance_ratio_[:n_pcs]
    sns.lineplot(x=range(n_pcs), y=np.cumsum(ve), ax=axarr[ix])
    axarr[ix].set_title('{} PCs, variance explained: {:.4f}'.format(
        n_pcs_list[ix], sum(ve, 0)))
    axarr[ix].set_xlabel('# of PCs')
    if ix == 0:
        axarr[ix].set_ylabel('Cumulative variance explained')
plt.suptitle('450k methylation data, # PCs vs. variance explained')
plt.subplots_adjust(top=0.85)

