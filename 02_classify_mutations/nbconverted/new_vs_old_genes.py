#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.data_utilities as du
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# location of classification results
results_dir = Path(cfg.results_dirs['mutation'], 'merged_all', 'gene').resolve()

# set significance cutoff after FDR correction
SIG_ALPHA = 0.001


# ### Load results and perform statistical testing/correction

# In[3]:


# load raw data
results_df = au.load_stratified_prediction_results(results_dir, 'gene')

# here we want to use compressed data for methylation datasets (27k and 450k)
# the results in 02_classify_compressed/compressed_vs_raw_results.ipynb show that
# performance is equal or slightly better for PCA compressed methylation data,
# and it's much easier/faster to fit models on
results_df = results_df[results_df.training_data.isin(['expression', 'rppa', 'mirna', 'mut_sigs'])]

# make sure that we're correctly pointing to raw data for non-methylation data types
# and that we have data for two replicates (two random seeds)
print(results_df.shape)
print(results_df.seed.unique())
print(results_df.training_data.unique())
results_df.head()


# In[4]:


# load compressed data for me_27k and me_450k
compressed_results_df = au.load_compressed_prediction_results(results_dir, 'gene')

# make sure that we're correctly pointing to compressed methylation data
# and that we have data for one dimension and two replicates (two random seeds)
print(compressed_results_df.shape)
print(compressed_results_df.seed.unique())
print(compressed_results_df.training_data.unique())
print(compressed_results_df.n_dims.unique())
compressed_results_df.head()


# In[5]:


results_df['n_dims'] = 'raw'
results_df = pd.concat((results_df, compressed_results_df))
print(results_df.seed.unique())
print(results_df.training_data.unique())
print(results_df.n_dims.unique())
print(results_df.shape)
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
results_df.training_data.replace(to_replace=training_data_map, inplace=True)
results_df.head()


# In[6]:


all_results_df = au.compare_all_data_types(results_df,
                                           SIG_ALPHA,
                                           metric='aupr')

cfg.sig_genes_dir.mkdir(exist_ok=True)
all_results_df.to_csv(cfg.sig_genes_all, index=False, sep='\t')

all_results_df.sort_values(by='p_value').head(10)


# ### Load gene sets

# In[7]:


# load datasets
vogelstein_df = du.load_vogelstein()
vogelstein_df.head()


# In[8]:


# load datasets
merged_df = du.load_merged()
merged_df.head()


# In[9]:


vogelstein_genes = set(vogelstein_df.gene)
non_vogelstein_genes = set(merged_df.gene) - vogelstein_genes
print(len(vogelstein_genes), len(non_vogelstein_genes))


# ### Plot results for Vogelstein and non-Vogelstein genes separately

# In[10]:


sns.set({'figure.figsize': (21, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(all_results_df[all_results_df.gene.isin(vogelstein_genes)],
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)

plt.suptitle('Results for Vogelstein genes only', size=18)
plt.tight_layout()


# In[11]:


sns.set({'figure.figsize': (21, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(all_results_df[all_results_df.gene.isin(non_vogelstein_genes)],
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)

plt.suptitle('Results for non-Vogelstein genes in merged gene set', size=18)
plt.tight_layout()


# ### Explore results for DNA damage repair genes

# In[12]:


# this list of genes in DDR pathways comes from Table S1 in Knijnenburg et al. 2018
# https://doi.org/10.1016/j.celrep.2018.03.076
ddr_genes_df = pd.read_excel(
    cfg.data_dir / 'mmc2.xlsx',
    skiprows = 3
)

print(ddr_genes_df.shape)
ddr_genes_df.head()


# In[13]:


ddr_genes = set(ddr_genes_df['Gene Symbol'])
ddr_vogelstein_genes = ddr_genes.intersection(vogelstein_genes)
print(len(ddr_vogelstein_genes))
print(ddr_vogelstein_genes)


# In[14]:


ddr_non_vogelstein_genes = ddr_genes.intersection(non_vogelstein_genes)
print(len(ddr_non_vogelstein_genes))
print(ddr_non_vogelstein_genes)


# In[15]:


print(ddr_vogelstein_genes.union(ddr_non_vogelstein_genes))
ddr_df = all_results_df[all_results_df.gene.isin(ddr_vogelstein_genes.union(
                                                  ddr_non_vogelstein_genes))]
print(ddr_df.gene.unique())


# In[16]:


sns.set({'figure.figsize': (21, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# plot mutation prediction from expression, in a volcano-like plot
datasets = ['gene expression', '27k methylation', '450k methylation']
filtered_data_map = {k: v for k, v in training_data_map.items() if v in datasets}

plu.plot_volcano_baseline(all_results_df[
                              all_results_df.gene.isin(ddr_vogelstein_genes.union(
                                                       ddr_non_vogelstein_genes))
                          ],
                          axarr,
                          filtered_data_map,
                          SIG_ALPHA,
                          metric='aupr',
                          verbose=True)

plt.suptitle('Results for DDR genes only', size=18)
plt.tight_layout()

