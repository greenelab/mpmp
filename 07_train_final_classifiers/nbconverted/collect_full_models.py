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

raw_data_types = [
    'expression',
    'rppa',
    'mirna',
    'mut_sigs'
]

compressed_data_types = [
    'me_27k',
    'me_450k'
]


# ### Get all possible -omics features
# 
# This will be the index for our coefficient dataframe. We'll use `NA` to denote features that weren't used in that particular model (e.g. a gene expression feature that wasn't in the top 8000 by MAD, or a cancer type indicator that wasn't included for that gene).

# In[3]:


def load_features(training_data, compressed=False, n_dim=5000):
    # get all cancer types
    sample_info_df = du.load_sample_info('expression')
    cancer_types = np.sort(sample_info_df.cancer_type.unique())
    
    # the columns will include the sample id, so use all but the first one
    if compressed:
        output_prefix = du.get_compress_output_prefix(
            training_data,
            n_dim,
            cfg.default_seed,
            standardize_input=True
        )
        omics_features = pd.read_csv(
            cfg.compressed_data_dir / '{}.tsv.gz'.format(output_prefix),
            sep='\t', nrows=0
        ).columns[1:].to_numpy()
    else:
        omics_features = pd.read_csv(
            cfg.data_types[training_data], sep='\t', nrows=0
        ).columns[1:].to_numpy()
    
    all_feats = np.concatenate((
        omics_features,
        cancer_types,
        np.array(['log10_mut'])
    ))
    return all_feats

exp_feats = load_features('expression')
print(exp_feats.shape)
print(exp_feats[:5], exp_feats[-5:])


# In[4]:


rppa_feats = load_features('rppa')
print(rppa_feats[:5], rppa_feats[-5:])


# In[5]:


me_27k_feats = load_features('me_27k', compressed=True)
print(me_27k_feats[:5], me_27k_feats[-5:])


# ### Load coefficients and assemble into dataframe

# In[6]:


def load_coefs(training_data, compressed=False):
    coefs = {}
    genes = []
    
    all_feats = load_features(training_data, compressed=compressed)

    # load coefficient vectors from output files, into dict
    for gene_dir in results_dir.iterdir():
        gene_name = gene_dir.stem
        gene_dir = Path(results_dir, gene_dir)
        if gene_dir.is_file(): continue
        genes.append(gene_name)
        for results_file in gene_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.stem)
            if training_data not in results_filename: continue
            if 'coefficients' not in results_filename: continue
            coefs_df = pd.read_csv(results_file, sep='\t')
            coefs[gene_name] = (coefs_df
                .loc[:, ['feature', 'weight']]
                .set_index('feature')
                # reindex will add NaN rows for features that weren't used,
                # this is what we want here
                .reindex(all_feats)
                .rename(columns={'weight': gene_name})
            )
            
    return coefs, genes
                    
coefs, genes = load_coefs('rppa')
print(genes[:5])
print(len(genes))


# In[7]:


# make sure all genes with parameters have classifiers
# the set difference should be empty
print(len(set(genes) - set(coefs.keys())))
print(set(genes) - set(coefs.keys()))


# In[8]:


assert len(set(genes) - set(coefs.keys())) == 0


# In[9]:


gene = 'PIK3CA'
print(coefs[gene].isna().sum())
coefs[gene].head()


# In[10]:


coefs[gene][coefs[gene][gene].isna()].head()


# In[11]:


na_feats = coefs[gene][coefs[gene][gene].isna()].index
print(len(na_feats), len(coefs[gene].index))
assert len(na_feats) < len(coefs[gene].index)


# In[12]:


# concatenate coefficient vectors into a single dataframe and save
(cfg.data_dir / 'final_models').mkdir(exist_ok=True)

# first do this for data types that use raw features
for data_type in raw_data_types:
    coefs, genes = load_coefs(data_type)
    coefs_df = (
        pd.concat(coefs.values(), axis='columns')
          .sort_index(axis='columns')
    )
    coefs_df.index.name = None
    # cfg.final_coefs_df should be a Path object, set in config.py
    output_fname = cfg.final_coefs_df.format(data_type)
    coefs_df.to_csv(output_fname, sep='\t')
    print(output_fname)


# In[13]:


# now handle data types that use compressed PCA features
for data_type in compressed_data_types:
    coefs, genes = load_coefs(data_type, compressed=True)
    coefs_df = (
        pd.concat(coefs.values(), axis='columns')
          .sort_index(axis='columns')
    )
    coefs_df.index.name = None
    # cfg.final_coefs_df should be a Path object, set in config.py
    output_fname = cfg.final_coefs_df.format(data_type)
    coefs_df.to_csv(output_fname, sep='\t')
    print(output_fname)


# ### Load parameters and assemble into dataframe

# In[14]:


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


# In[15]:


params[gene].head()


# In[16]:


# concatenate lists of selected parameters into a single dataframe
params_df = (
    pd.concat(params.values(), axis='rows')
      .sort_index(axis='rows')
)

print(params_df.shape)
params_df.iloc[:5, :5]


# In[17]:


# cfg.final_params_df should be a Path object, set in config.py
params_df.to_csv(cfg.final_params_df, sep='\t')
print(cfg.final_params_df)

