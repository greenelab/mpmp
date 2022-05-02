#!/usr/bin/env python
# coding: utf-8

# ## Collect full models and save into tsv files
# 
# As a resource to provide with our paper, we want to provide the fit models (coefficients/effect sizes, and parameter choices) as a data file.
# 
# This notebook collects the results of the bash script at `07_train_final_classifiers/scripts/run_all_genes.sh` and assembles them into dataframes/`.tsv` files.

# In[1]:


from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = Path(cfg.results_dirs['final'],
                   'merged_all_params',
                   'gene').resolve()


# ### Get all possible gene expression features
# 
# This will be the index for our coefficient dataframe. We'll use `NA` to denote features that weren't used in that particular model (either a gene expression feature that wasn't in the top 8000 by MAD, or a cancer type indicator that wasn't included for that gene).

# In[3]:


# get all cancer types
sample_info_df = du.load_sample_info('expression')
cancer_types = np.sort(sample_info_df.cancer_type.unique())
print(cancer_types.shape)
print(cancer_types)


# In[4]:


# the columns will include the sample id, so use all but the first one
gene_features = pd.read_csv(
    cfg.data_types['expression'], sep='\t', nrows=0
).columns[1:].values

print(gene_features.shape)
gene_features[:5]


# In[5]:


all_feats = np.concatenate((
    gene_features,
    cancer_types,
    np.array(['log10_mut'])
))
print(all_feats.shape)


# ### Load coefficients and assemble into dataframe

# In[6]:


coefs = {}
genes = []

# load coefficient vectors from output files, into dict
for gene_dir in results_dir.iterdir():
    gene_name = gene_dir.stem
    gene_dir = Path(results_dir, gene_dir)
    if gene_dir.is_file(): continue
    genes.append(gene_name)
    for results_file in gene_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if 'coefficients' not in results_filename: continue
        coefs_df = pd.read_csv(results_file, sep='\t')
        coefs[gene_name] = (coefs_df
            .loc[:, ['feature', 'weight']]
            .set_index('feature')
            .reindex(all_feats)
            .rename(columns={'weight': gene_name})
        )
                    
print(genes[:5])
print(len(genes))


# In[7]:


# make sure all genes with parameters have classifiers
# the set difference should be empty
print(len(set(genes) - set(coefs.keys())))
print(set(genes) - set(coefs.keys()))


# In[8]:


gene = 'PIK3CA'
print(coefs[gene].isna().sum())
coefs[gene].head()


# In[9]:


coefs[gene][coefs[gene][gene].isna()].head()


# In[10]:


# concatenate coefficient vectors into a single dataframe
coefs_df = (
    pd.concat(coefs.values(), axis='columns')
      .sort_index(axis='columns')
)
coefs_df.index.name = None

print(coefs_df.shape)
coefs_df.iloc[:5, :5]


# In[11]:


(cfg.data_dir / 'final_models').mkdir(exist_ok=True)
coefs_df.to_csv(cfg.final_coefs_df, sep='\t')


# ### Load parameters and assemble into dataframe

# In[12]:


params = {}

# load parameter lists from output files, into dict
for gene_dir in results_dir.iterdir():
    gene_name = gene_dir.stem
    gene_dir = Path(results_dir, gene_dir)
    if gene_dir.is_file(): continue
    for results_file in gene_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if 'params' not in results_filename: continue
        with open(results_file, 'rb') as f:
            gene_params = pkl.load(f)
        params[gene_name] = pd.DataFrame(
            gene_params, index=[gene_name]
        )
        
print(list(params.keys())[:5])
print(len(params.keys()))


# In[13]:


params[gene].head()


# In[14]:


# concatenate lists of selected parameters into a single dataframe
params_df = (
    pd.concat(params.values(), axis='rows')
      .sort_index(axis='rows')
)

print(params_df.shape)
params_df.iloc[:5, :5]


# In[15]:


params_df.to_csv(cfg.final_params_df, sep='\t')

