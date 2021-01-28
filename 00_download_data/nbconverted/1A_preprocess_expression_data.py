#!/usr/bin/env python
# coding: utf-8

# ## Preprocess pan-cancer gene expression data

# Load the downloaded data, update gene identifiers to Entrez, and curate sample IDs.

# In[1]:


import os
import pandas as pd

import mpmp.config as cfg


# ### Read TCGA Barcode Curation Information
# 
# Extract information from TCGA barcodes - `cancer-type` and `sample-type`. See https://github.com/cognoma/cancer-data for more details

# In[2]:


# commit from https://github.com/cognoma/cancer-data/
sample_commit = 'da832c5edc1ca4d3f665b038d15b19fced724f4c'
url = 'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_cancertype_codes.csv'.format(sample_commit)
cancer_types_df = pd.read_csv(url,
                              dtype='str',
                              keep_default_na=False)

cancertype_codes_dict = dict(zip(cancer_types_df['TSS Code'],
                                 cancer_types_df.acronym))
cancer_types_df.head(2)


# In[3]:


url = 'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_sampletype_codes.csv'.format(sample_commit)
sample_types_df = pd.read_csv(url, dtype='str')

sampletype_codes_dict = dict(zip(sample_types_df.Code,
                                 sample_types_df.Definition))
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

tcga_expr_df.to_csv(cfg.rnaseq_data, sep='\t', compression='gzip', float_format='%.3g')
# In[13]:


print(tcga_expr_df.shape)
tcga_expr_df.head()


# ### Process TCGA cancer-type and sample-type info from barcodes
# 
# Cancer-type includes `OV`, `BRCA`, `LUSC`, `LUAD`, etc. while sample-type includes `Primary`, `Metastatic`, `Solid Tissue Normal`, etc.
# 
# See https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes for more details.
# 
# The goal is to use this info to stratify training (90%) and testing (10%) balanced by cancer-type and sample-type. 

# In[14]:


# extract sample type in the order of the gene expression matrix
tcga_id = pd.DataFrame(tcga_expr_df.index)

# extract the last two digits of the barcode and recode sample-type
tcga_id = tcga_id.assign(sample_type = tcga_id.sample_id.str[-2:])
tcga_id.sample_type = tcga_id.sample_type.replace(sampletype_codes_dict)

# extract the first two ID numbers after `TCGA-` and recode cancer-type
tcga_id = tcga_id.assign(cancer_type = tcga_id.sample_id.str[5:7])
tcga_id.cancer_type = tcga_id.cancer_type.replace(cancertype_codes_dict)

# append cancer-type with sample-type to generate stratification variable
tcga_id = tcga_id.assign(id_for_stratification = tcga_id.cancer_type.str.cat(tcga_id.sample_type))

# get stratification counts - function cannot work with singleton strats
stratify_counts = tcga_id.id_for_stratification.value_counts().to_dict()

# recode stratification variables if they are singletons
tcga_id = tcga_id.assign(stratify_samples_count = tcga_id.id_for_stratification)
tcga_id.stratify_samples_count = tcga_id.stratify_samples_count.replace(stratify_counts)
tcga_id.loc[tcga_id.stratify_samples_count == 1, "stratify_samples"] = "other"


# In[15]:


# write out files for downstream use
file = os.path.join(cfg.data_dir, 'tcga_sample_identifiers.tsv')

(
    tcga_id.drop(['stratify_samples', 'stratify_samples_count'], axis='columns')
    .to_csv(file, sep='\t', index=False)
)

print(tcga_id.shape)
tcga_id.head()


# In[16]:


cancertype_count_df = (
    pd.DataFrame(tcga_id.cancer_type.value_counts())
    .reset_index()
    .rename({'index': 'cancertype', 'cancer_type': 'n ='}, axis='columns')
)

file = os.path.join(cfg.data_dir, 'tcga_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)

cancertype_count_df.head()


# In[17]:


# take PCA + save to file, for equal comparison with methylation
from sklearn.decomposition import PCA

pca_dir = os.path.join(cfg.data_dir, 'exp_compressed')
os.makedirs(pca_dir, exist_ok=True)

n_pcs_list = [100, 1000, 5000]
for n_pcs in n_pcs_list:
    pca = PCA(n_components=n_pcs)
    exp_pca = pca.fit_transform(tcga_expr_df)
    print(exp_pca.shape)
    exp_pca = pd.DataFrame(exp_pca, index=tcga_expr_df.index)
    exp_pca.to_csv(os.path.join(pca_dir,
                               'exp_pc{}.tsv.gz'.format(n_pcs)),
                   sep='\t',
                   float_format='%.3g')

