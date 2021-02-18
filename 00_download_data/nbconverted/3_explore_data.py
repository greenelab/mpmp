#!/usr/bin/env python
# coding: utf-8

# ### Explore processed pan-cancer data

# In[1]:


import os
import sys

import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du


# In[2]:


# load gene/classification info and sample/cancer type info
print('Loading gene label data...', file=sys.stderr)
genes_df = du.load_vogelstein()
sample_info_df = du.load_sample_info(verbose=True)

# load mutation info
# this returns a tuple of dataframes, unpack it below
pancancer_data = du.load_pancancer_data(verbose=True)
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data


# In[3]:


# load expression data
rnaseq_df = du.load_expression_data(verbose=True)

# standardize columns of expression dataframe
print('Standardizing columns of expression data...', file=sys.stderr)
rnaseq_df[rnaseq_df.columns] = StandardScaler().fit_transform(rnaseq_df[rnaseq_df.columns])

print(rnaseq_df.shape)
rnaseq_df.iloc[:5, :5]


# In[4]:


# load methylation data
methylation_df = du.load_methylation_data(verbose=True)

# standardize columns of methylation dataframe
# decided not to do this for now, unnormalized beta values seem to be informative
# print('Standardizing columns of methylation data...', file=sys.stderr)
# methylation_df[methylation_df.columns] = StandardScaler().fit_transform(
#     methylation_df[methylation_df.columns])

print(methylation_df.shape)
methylation_df.iloc[:5, :5]


# First, let's compare low-dimensional representations of gene expression and DNA methylation data. These representations should be somewhat similar between data types (DNA methylation affects/correlates with gene expression), and we should see a decent separation between cancer types in the plots.
# 
# We'll choose a few cancer types that are similar to one another (LUSC/LUAD, LGG/GBM) and a few that should be dissimilar (BRCA, THCA).

# In[5]:


from sklearn.decomposition import PCA
from umap import UMAP

sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)

assert sample_info_df.index.equals(rnaseq_df.index)

pca = PCA(n_components=2)
X_proj_pca = pca.fit_transform(rnaseq_df)
reducer = UMAP(n_components=2, random_state=42)
X_proj_umap = reducer.fit_transform(rnaseq_df)

enum_samples_df = sample_info_df.reset_index()
rnaseq_cancer_types = sorted(sample_info_df.cancer_type.unique())
# rnaseq_cancer_types = ['LUAD', 'LUSC', 'THCA', 'LGG', 'GBM', 'BRCA']
for i, cancer_type in enumerate(rnaseq_cancer_types):
    ixs = enum_samples_df.index[enum_samples_df['cancer_type'] == cancer_type].tolist()
    axarr[0].scatter(X_proj_pca[ixs, 0], X_proj_pca[ixs, 1], label=cancer_type, s=3)
    axarr[1].scatter(X_proj_umap[ixs, 0], X_proj_umap[ixs, 1], label=cancer_type, s=3)
    
axarr[0].set_xlabel('PC1')
axarr[0].set_ylabel('PC2')
axarr[0].set_title('PCA projection of TCGA expression data, colored by cancer type')
axarr[0].legend()
axarr[1].set_xlabel('UMAP dimension 1')
axarr[1].set_ylabel('UMAP dimension 2')
axarr[1].set_title('UMAP projection of TCGA expression data, colored by cancer type')
axarr[1].legend()


# In[6]:


sample_overlap = rnaseq_df.index.intersection(methylation_df.index)
print(len(rnaseq_df), len(methylation_df), len(sample_overlap))


# In[7]:


sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)

methylation_filtered_df = methylation_df.reindex(sample_overlap)
enum_samples_df = (
    sample_info_df.reindex(sample_overlap)
                  .reset_index()
)

pca = PCA(n_components=2)
X_proj_pca = pca.fit_transform(methylation_filtered_df)
reducer = UMAP(n_components=2, random_state=42)
X_proj_umap = reducer.fit_transform(methylation_filtered_df)

me_cancer_types = sorted(sample_info_df.cancer_type.unique())
# me_cancer_types = ['LUAD', 'LUSC', 'THCA', 'LGG', 'GBM', 'BRCA']
for i, cancer_type in enumerate(me_cancer_types):
    ixs = enum_samples_df.index[enum_samples_df['cancer_type'] == cancer_type].tolist()
    axarr[0].scatter(X_proj_pca[ixs, 0], X_proj_pca[ixs, 1], label=cancer_type, s=3)
    axarr[1].scatter(X_proj_umap[ixs, 0], X_proj_umap[ixs, 1], label=cancer_type, s=3)
    
axarr[0].set_xlabel('PC1')
axarr[0].set_ylabel('PC2')
axarr[0].set_title('PCA projection of TCGA methylation data, colored by cancer type')
axarr[0].legend()
axarr[1].set_xlabel('UMAP dimension 1')
axarr[1].set_ylabel('UMAP dimension 2')
axarr[1].set_title('UMAP projection of TCGA methylation data, colored by cancer type')
axarr[1].legend()


# Now we want to dig a bit deeper into LGG and GBM. It's fairly well-known that IDH1 mutation status defines distinct subtypes in both classes of brain tumors. We'll compare methylation and gene expression in IDH1-mutated vs. non-mutated samples, expecting to see a separation in our low dimensional representation.
# 
# IDH1 plays a direct role in DNA methylation, so we anticipate that this separation between mutated and non-mutated samples will be slightly clearer in the methylation data.

# In[8]:


from mpmp.utilities.tcga_utilities import process_y_matrix

def generate_labels(gene, classification):
    # process the y matrix for the given gene or pathway
    y_mutation_df = mutation_df.loc[:, gene]

    # include copy number gains for oncogenes
    # and copy number loss for tumor suppressor genes (TSG)
    include_copy = True
    if classification == "Oncogene":
        y_copy_number_df = copy_gain_df.loc[:, gene]
    elif classification == "TSG":
        y_copy_number_df = copy_loss_df.loc[:, gene]
    else:
        y_copy_number_df = pd.DataFrame()
        include_copy = False

    # construct labels from mutation/CNV information, and filter for
    # cancer types without an extreme label imbalance
    y_df = process_y_matrix(
        y_mutation=y_mutation_df,
        y_copy=y_copy_number_df,
        include_copy=include_copy,
        gene=gene,
        sample_freeze=sample_freeze_df,
        mutation_burden=mut_burden_df,
        filter_count=1,
        filter_prop=0.01,
        output_directory=None,
        hyper_filter=5,
        test=True # don't write filter info to file
    )
    return y_df


# In[9]:


gene = 'IDH1'
cancer_types = ['LGG', 'GBM']
classification = du.get_classification(gene, genes_df)
y_df = generate_labels(gene, classification)

y_df = y_df[y_df.DISEASE.isin(cancer_types)]
print(y_df.shape)
y_df.tail()


# In[10]:


# generate UMAP 2-dimensional representations of data
shuffle = False

def shuffle_cols(input_df):
    # randomly permute genes of each sample in the rnaseq matrix
    shuf_df = input_df.apply(lambda x:
                             np.random.permutation(x.tolist()),
                             axis=1)
    # set up new dataframe
    shuf_df = pd.DataFrame(shuf_df, columns=['col_list'])
    shuf_df = pd.DataFrame(shuf_df.col_list.values.tolist(),
                           columns=input_df.columns,
                           index=input_df.index)
    return shuf_df

# get samples that are present in all 3 datasets (expression, methylation, mutations)
ix_overlap = y_df.index.intersection(rnaseq_df.index).intersection(methylation_filtered_df.index)
y_mut_df = y_df.loc[ix_overlap, :]
rnaseq_mut_df = rnaseq_df.loc[ix_overlap, :]
me_mut_df = methylation_filtered_df.loc[ix_overlap, :]

if shuffle:
    rnaseq_mut_df = shuffle_cols(rnaseq_mut_df)
    me_mut_df = shuffle_cols(me_mut_df)
    
reducer = UMAP(n_components=2, random_state=42)
X_proj_rnaseq = reducer.fit_transform(rnaseq_mut_df)
X_proj_me = reducer.fit_transform(me_mut_df)
print(X_proj_rnaseq.shape)
print(X_proj_me.shape)


# In[11]:


gene_label = '{} mutant'.format(gene)
me_proj_df = pd.DataFrame({
    'UMAP1': X_proj_me[:, 0],
    'UMAP2': X_proj_me[:, 1],
    'Cancer type': y_mut_df.DISEASE.values,
    gene_label: y_mut_df.status.values.astype('bool')
})
rnaseq_proj_df = pd.DataFrame({
    'UMAP1': X_proj_rnaseq[:, 0],
    'UMAP2': X_proj_rnaseq[:, 1],
    'Cancer type': y_mut_df.DISEASE.values,
    gene_label: y_mut_df.status.values.astype('bool')
})
me_proj_df.head()


# In[12]:


sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)

sns.scatterplot(x='UMAP1', y='UMAP2', data=me_proj_df, hue=gene_label,
                style='Cancer type', ax=axarr[0])
axarr[0].set_xlabel('UMAP dimension 1')
axarr[0].set_ylabel('UMAP dimension 2')
axarr[0].set_title('UMAP projection of TCGA methylation data, colored by mutation status')
axarr[0].legend()
sns.scatterplot(x='UMAP1', y='UMAP2', data=rnaseq_proj_df, hue=gene_label,
                style='Cancer type', ax=axarr[1])
axarr[1].set_xlabel('UMAP dimension 1')
axarr[1].set_ylabel('UMAP dimension 2')
axarr[1].set_title('UMAP projection of TCGA gene expression data, colored by mutation status')
axarr[1].legend()


# As expected, we can see that there's a nice separation between (most) IDH1 mutants and non-mutants in the methylation data. They separate to some degree in the gene expression data, but not quite as clearly.
# 
# It's likely (although I haven't checked this yet) that the non-mutated samples in the IDH1-mutant methylation cluster are actually IDH2 mutants. IDH2 is thought to phenocopy IDH1 in gliomas, having a similar effect on methylation and gene expression as IDH1 when mutated.
