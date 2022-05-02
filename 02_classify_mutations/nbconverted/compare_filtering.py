#!/usr/bin/env python
# coding: utf-8

# ## Compare dataset filtering methods
# 
# Instead of applying our mutation count/percentage filters to samples from each cancer type independently, we could just look at the whole dataset - or in other words, we could choose genes to train classifiers for based on their overall percent/count of mutated samples.
# 
# This script compares the per-cancer type filtering method (or the "cancer_type" method) to the per-gene, across cancer types filtering method (or the "gene" method).

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


cancer_type_results_dir = Path(cfg.results_dirs['mutation'],
                               'merged_all',
                               'gene').resolve()

gene_results_dir = Path(cfg.results_dirs['mutation'],
                        'merged_filter_all',
                        'gene').resolve()

# set significance cutoff after FDR correction
SIG_ALPHA = 0.001


# ### Plot results with new filtering scheme

# In[3]:


# load raw data
gene_results_df = au.load_stratified_prediction_results(gene_results_dir, 'gene')

# here we want to use compressed data for methylation datasets (27k and 450k)
# the results in 02_classify_compressed/compressed_vs_raw_results.ipynb show that
# performance is equal or slightly better for PCA compressed methylation data,
# and it's much easier/faster to fit models on
gene_results_df = gene_results_df[gene_results_df.training_data.isin(['expression', 'rppa', 'mirna', 'mut_sigs'])]

# make sure that we're correctly pointing to raw data for non-methylation data types
# and that we have data for two replicates (two random seeds)
print(gene_results_df.shape)
print(gene_results_df.seed.unique())
print(gene_results_df.training_data.unique())
gene_results_df.head()


# In[4]:


# load compressed data for me_27k and me_450k
gene_compressed_results_df = au.load_compressed_prediction_results(
    gene_results_dir, 'gene')

# make sure that we're correctly pointing to compressed methylation data
# and that we have data for one dimension and two replicates (two random seeds)
print(gene_compressed_results_df.shape)
print(gene_compressed_results_df.seed.unique())
print(gene_compressed_results_df.training_data.unique())
print(gene_compressed_results_df.n_dims.unique())
gene_compressed_results_df.head()


# In[5]:


gene_results_df['n_dims'] = 'raw'
gene_results_df = pd.concat((gene_results_df, gene_compressed_results_df))
print(gene_results_df.seed.unique())
print(gene_results_df.training_data.unique())
print(gene_results_df.n_dims.unique())
print(gene_results_df.shape)
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
gene_results_df.training_data.replace(to_replace=training_data_map, inplace=True)
gene_results_df.head()


# In[6]:


gene_all_results_df = au.compare_all_data_types(gene_results_df,
                                               SIG_ALPHA,
                                               metric='aupr')

gene_all_results_df.sort_values(by='p_value').head(10)


# In[7]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(gene_all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)


# In[8]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['RPPA', 'microRNA', 'mutational signatures']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(gene_all_results_df,
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)


# ### Compare against results with old filtering scheme

# In[9]:


# load raw data
cancer_type_results_df = au.load_stratified_prediction_results(cancer_type_results_dir, 'gene')

# here we want to use compressed data for methylation datasets (27k and 450k)
# the results in 02_classify_compressed/compressed_vs_raw_results.ipynb show that
# performance is equal or slightly better for PCA compressed methylation data,
# and it's much easier/faster to fit models on
cancer_type_results_df = cancer_type_results_df[cancer_type_results_df.training_data.isin(['expression', 'rppa', 'mirna', 'mut_sigs'])]

# make sure that we're correctly pointing to raw data for non-methylation data types
# and that we have data for two replicates (two random seeds)
print(cancer_type_results_df.shape)
print(cancer_type_results_df.seed.unique())
print(cancer_type_results_df.training_data.unique())
cancer_type_results_df.head()


# In[10]:


# load compressed data for me_27k and me_450k
cancer_type_compressed_results_df = au.load_compressed_prediction_results(
    cancer_type_results_dir, 'gene')

# make sure that we're correctly pointing to compressed methylation data
# and that we have data for one dimension and two replicates (two random seeds)
print(cancer_type_compressed_results_df.shape)
print(cancer_type_compressed_results_df.seed.unique())
print(cancer_type_compressed_results_df.training_data.unique())
print(cancer_type_compressed_results_df.n_dims.unique())
cancer_type_compressed_results_df.head()


# In[11]:


cancer_type_results_df['n_dims'] = 'raw'
cancer_type_results_df = pd.concat((cancer_type_results_df, cancer_type_compressed_results_df))
print(cancer_type_results_df.seed.unique())
print(cancer_type_results_df.training_data.unique())
print(cancer_type_results_df.n_dims.unique())
print(cancer_type_results_df.shape)
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
cancer_type_results_df.training_data.replace(to_replace=training_data_map, inplace=True)
cancer_type_results_df.head()


# In[12]:


cancer_type_all_results_df = au.compare_all_data_types(cancer_type_results_df,
                                               SIG_ALPHA,
                                               metric='aupr')

cancer_type_all_results_df.sort_values(by='p_value').head(10)


# In[13]:


overlap_genes = (
    set(gene_all_results_df.gene.unique())
      .intersection(set(cancer_type_all_results_df.gene.unique()))
)

gene_all_results_df = gene_all_results_df[gene_all_results_df.gene.isin(overlap_genes)]
cancer_type_all_results_df = cancer_type_all_results_df[cancer_type_all_results_df.gene.isin(overlap_genes)]

cols = ['gene', 'training_data', 'delta_mean']
diff_df = (cancer_type_all_results_df
  .loc[:, cols]
  .merge(gene_all_results_df.loc[:, cols],
         left_on=['gene', 'training_data'],
         right_on=['gene', 'training_data'])
)

diff_df.rename(
    columns={'delta_mean_x': 'delta_mean_cancer_type',
             'delta_mean_y': 'delta_mean_gene'},
    inplace=True
)
diff_df['cancer_type_vs_gene'] = diff_df.delta_mean_cancer_type - diff_df.delta_mean_gene
diff_df.head()


# In[14]:


sns.set({'figure.figsize': (12, 6)})

sns.boxplot(data=diff_df, x='training_data', y='cancer_type_vs_gene',
            order=training_data_map.values())
plt.xlabel('Training data')
plt.ylabel('(Cancer type - gene) filtering scheme')


# In[15]:


metric = 'aupr'
num_examples = 20

top_df = (diff_df
    .sort_values(by='cancer_type_vs_gene', ascending=False)
    .head(num_examples)
)
bottom_df = (diff_df
    .sort_values(by='cancer_type_vs_gene', ascending=False)
    .tail(num_examples)
)

plot_df = pd.concat((top_df, bottom_df)).reset_index()

sns.set({'figure.figsize': (30, 5)})
sns.barplot(data=plot_df, x=plot_df.index, y='cancer_type_vs_gene',
            hue='training_data', dodge=False,
            hue_order=training_data_map.values())

def show_values_on_bars(ax):
    for i in range(plot_df.shape[0]):
        _x = i
        _y = plot_df.loc[i, 'cancer_type_vs_gene']
        val = plot_df.loc[i, 'gene']
        if _y > 0:
            ax.text(_x, _y + 0.02, val, ha="center") 
        else:
            ax.text(_x, _y - 0.04, val, ha="center") 
        
show_values_on_bars(plt.gca())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel('AUPR difference')
plt.title('Performance difference between cancer type and gene filtering scheme', size=14)


# The plots above show the difference for each gene between the "old" filtering scheme (per-cancer type) and the "new" filtering scheme (gene-level across all cancer types). A positive value for a gene means that gene's classifier performed better for the "old" filtering scheme and vice-versa for the "new" filtering scheme.
# 
# In the box plot, we can see that most genes have a positive old vs. new difference (means > 0 and small negative tails), indicating that filtering for each cancer type independently (the "old" approach) tends to lead to better classifier performance for most genes. This seems to hold across all the data types we looked at.
# 
# The bar plot shows the extremes on each end (i.e. genes with the largest negative or positive difference). The positive tails tend to be larger than the negative tails, supporting our observations from the box plot.