#!/usr/bin/env python
# coding: utf-8

# ## Use pre-trained models to make predictions on normal tissue samples
# 
# For some cancer types, TCGA provides samples from normal tissue in addition to the tumor samples (see `01_explore_data/normal_tissue_samples.ipynb`).
# 
# In this analysis, we want to make predictions on those samples and compare them to our tumor sample predictions.
# 
# Our assumption is that our models will predict that the normal tissue samples have a low probability of mutation (since they almost certainly do not have somatic mutations in any of the genes of interest).

# In[1]:


from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.data_utilities as du
import mpmp.utilities.plot_utilities as plu
import mpmp.utilities.tcga_utilities as tu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


results_dir = Path(cfg.results_dirs['final'],
                   'pilot_genes',
                   'gene').resolve()

genes = [g.stem for g in results_dir.iterdir() if not g.is_file()]
print(genes)


# ### Load pre-trained model

# In[3]:


gene = 'TP53'
training_data = 'expression'

model_filename = '{}_{}_elasticnet_classify_s42_model.pkl'.format(gene, training_data)

with open(str(results_dir / gene / model_filename), 'rb') as f:
    model_fit = pkl.load(f)

print(model_fit)
print(model_fit.feature_names_in_.shape)


# ### Load expression data and sample info

# In[4]:


# load expression sample info, this has tumor/normal labels
sample_info_df = du.load_sample_info(training_data)
print(sample_info_df.sample_type.unique())
sample_info_df.head()


# In[5]:


# load expression data
data_df = du.load_raw_data('expression', verbose=True)
print(data_df.shape)
data_df.iloc[:5, :5]


# In[6]:


normal_ids = (
    sample_info_df[sample_info_df.sample_type.str.contains('Normal')]
      .index
      .intersection(data_df.index)
)
print(len(normal_ids))
print(normal_ids[:5])


# In[7]:


normal_data_df = data_df.loc[normal_ids, :]
print(normal_data_df.shape)
normal_data_df.iloc[:5, :5]


# ### Preprocessing for normal samples
# 
# This is a bit nuanced since we don't have mutation calling information for the normal samples, so we can't generate a log(mutation burden) covariate.
# 
# For now we'll just take the mean mutation burden from the tumor dataset and apply it to all the normal samples.

# In[8]:


# load mutation data
pancancer_data = du.load_pancancer_data()
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data


# In[9]:


print(mut_burden_df.shape)
mut_burden_df.head()


# In[10]:


mean_mutation_burden = mut_burden_df.sum() / mut_burden_df.shape[0]
print(mean_mutation_burden)


# In[11]:


# construct covariate matrix for normal samples
normal_info_df = pd.DataFrame(
    {'log10_mut': mean_mutation_burden.values[0]},
    index=normal_ids
)
# add cancer type
# TODO: this needs to use the same dummies as the training data,
# how to do that?
# 1) if normal samples are not in the set of cancer types the model
#    was trained on, drop them
# 2) if normal samples are in the set of cancer types the model
#    was trained on, set the dummies *the same way*
normal_info_df = (normal_info_df
    .merge(sample_info_df, left_index=True, right_index=True)    
    .drop(columns={'id_for_stratification'})
    .rename(columns={'cancer_type': 'DISEASE'})
)
print(normal_info_df.shape)
normal_info_df.head()


# In[12]:


def add_dummies_from_model(data_df, info_df, model):
    # get cancer type covariates used in original model,
    # in the correct order
    valid_cancer_types = sample_info_df.cancer_type.unique()
    cancer_type_covs = [
        f for f in model.feature_names_in_ if f in valid_cancer_types
    ]
    cov_matrix = np.zeros((info_df.shape[0], len(cancer_type_covs)))
    for sample_ix, (_, row) in enumerate(info_df.iterrows()):
        try:
            row_cancer_type = row.DISEASE
            cov_ix = cancer_type_covs.index(row_cancer_type)
            cov_matrix[sample_ix, cov_ix] = 1
        except ValueError:
            # if cancer type is not in model,
            # just leave it an all-zeros row
            continue
    mut_burden = info_df.log10_mut.values.reshape(-1, 1)
    feature_matrix = np.concatenate(
        (data_df, mut_burden, cov_matrix),
        axis=1
    )
    X_df = pd.DataFrame(
        feature_matrix,
        index=data_df.index.copy(),
        columns=model.feature_names_in_[:]
    )
    return X_df
    
X_df = add_dummies_from_model(normal_data_df, normal_info_df, model_fit)
print(X_df.shape)
X_df.iloc[:5, -20:]

