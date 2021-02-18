#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer gene expression data

# Load the downloaded data, update gene identifiers to Entrez, and curate sample IDs.

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


# ### Read Entrez ID Curation Information
# 
# Load curated gene names from versioned resource. See https://github.com/cognoma/genes for more details

# In[4]:


# commit from https://github.com/cognoma/genes
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'


# In[5]:


url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
gene_df = pd.read_csv(url, sep='\t')

# only consider protein-coding genes
gene_df = (
    gene_df.query("gene_type == 'protein-coding'")
)

print(gene_df.shape)
gene_df.head(2)


# In[6]:


# load gene updater - define up to date Entrez gene identifiers where appropriate
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
updater_df = pd.read_csv(url, sep='\t')

old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                             updater_df.new_entrez_gene_id))


# ### Load and process gene expression data

# In[7]:


# first load manifest file, this tells us the filenames of the raw data files
manifest_df = pd.read_csv(os.path.join(cfg.data_dir, 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head(1)


# In[8]:


tcga_expr_df = pd.read_csv(
    os.path.join(cfg.raw_data_dir, manifest_df.loc['rna_seq'].filename),
    index_col=0, sep='\t')

print(tcga_expr_df.shape)
tcga_expr_df.head()


# ### Process gene expression matrix
# 
# This involves updating Entrez gene ids, sorting and subsetting.

# In[9]:


# set index as entrez_gene_id
tcga_expr_df.index = tcga_expr_df.index.map(lambda x: x.split('|')[1])


# In[10]:


tcga_expr_df = (tcga_expr_df
    .dropna(axis='rows')
    .rename(index=old_to_new_entrez)
    .groupby(level=0).mean()
    .transpose()
    .sort_index(axis='rows')
    .sort_index(axis='columns')
)

tcga_expr_df.index.rename('sample_id', inplace=True)


# In[11]:


# update sample IDs to remove multiple samples measured on the same tumor
# and to map with the clinical information
tcga_expr_df.index = tcga_expr_df.index.str.slice(start=0, stop=15)
tcga_expr_df = tcga_expr_df.loc[~tcga_expr_df.index.duplicated(), :]


# In[12]:


# filter for valid Entrez gene identifiers
tcga_expr_df = tcga_expr_df.loc[:, tcga_expr_df.columns.isin(gene_df.entrez_gene_id.astype(str))]


# In[13]:


tcga_expr_df.to_csv(cfg.rnaseq_data, sep='\t', compression='gzip', float_format='%.3g')


# In[14]:


print(tcga_expr_df.shape)
tcga_expr_df.head()


# ### Process TCGA cancer type and sample type info from barcodes
# 
# Cancer-type includes `OV`, `BRCA`, `LUSC`, `LUAD`, etc. while sample-type includes `Primary`, `Metastatic`, `Solid Tissue Normal`, etc.
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.
# 
# The goal is to use this info to stratify train and test sets by cancer type and sample type. 

# In[15]:


# get sample info and save to file
tcga_id = tu.get_and_save_sample_info(tcga_expr_df,
                                      sampletype_codes_dict,
                                      cancertype_codes_dict)

print(tcga_id.shape)
tcga_id.head()


# In[15]:


# get cancer type counts and save to file
cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.sample_info_dir, 'tcga_expression_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()


# ### Dimension reduction
# 
# Compress the data using PCA with various dimensions, and save the results to tsv files.

# In[18]:


# take PCA + save to file, for equal comparison with methylation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# standardize first for expression data
tcga_scaled_df = pd.DataFrame(
    StandardScaler().fit_transform(tcga_expr_df),
    index=tcga_expr_df.index.copy(),
    columns=tcga_expr_df.columns.copy(),
)

pca_dir = os.path.join(cfg.data_dir, 'exp_compressed')
os.makedirs(pca_dir, exist_ok=True)

n_pcs_list = [100, 1000, 5000]
var_exp_list = []
for n_pcs in n_pcs_list:
    pca = PCA(n_components=n_pcs, random_state=cfg.default_seed)
    exp_pca = pca.fit_transform(tcga_scaled_df)
    print(exp_pca.shape)
    var_exp_list.append(pca.explained_variance_ratio_)
    exp_pca = pd.DataFrame(exp_pca, index=tcga_scaled_df.index)
    exp_pca.to_csv(os.path.join(pca_dir,
                               'exp_std_pc{}.tsv.gz'.format(n_pcs)),
                   sep='\t',
                   float_format='%.3g')


# In[19]:


# plot PCA variance explained

sns.set({'figure.figsize': (15, 4)})
fig, axarr = plt.subplots(1, 3)

for ix, ve in enumerate(var_exp_list):
    sns.lineplot(x=range(len(ve)), y=np.cumsum(ve), ax=axarr[ix])
    axarr[ix].set_title('{} PCs, variance explained: {:.4f}'.format(
        n_pcs_list[ix], sum(ve, 0)))
    axarr[ix].set_xlabel('# of PCs')
    if ix == 0:
        axarr[ix].set_ylabel('Cumulative variance explained')
plt.suptitle('Expression data, # PCs vs. variance explained')
plt.subplots_adjust(top=0.85)

