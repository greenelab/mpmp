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


# load mutation data
pancancer_data = du.load_pancancer_data()
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancancer_data


# ### Subset expression data to get train and control samples
# 
# We want to compare predictions made using the trained model on the control samples to predictions on the (tumor-derived) data used to train the model, so we'll load both expression datasets here.

# In[7]:


# get cancer types that were used to train the model
valid_cancer_types = sample_info_df.cancer_type.unique()
train_cancer_types = [
    f for f in model_fit.feature_names_in_ if f in valid_cancer_types
]
print(train_cancer_types)


# In[8]:


# get samples that were used to train the model
train_samples = (
    sample_info_df[sample_info_df.cancer_type.isin(train_cancer_types)]
      .index
      .intersection(data_df.index)
      .intersection(mut_burden_df.index)
)
train_data_df = data_df.loc[train_samples, :]
print(train_data_df.shape)
train_data_df.iloc[:5, :5]


# In[9]:


# get normal samples that we have expression data for
normal_ids = (
    sample_info_df[sample_info_df.sample_type.str.contains('Normal')]
      .index
      .intersection(data_df.index)
)
print(len(normal_ids))
print(normal_ids[:5])


# In[10]:


# get normal expression data
normal_data_df = data_df.loc[normal_ids, :]
print(normal_data_df.shape)
normal_data_df.iloc[:5, :5]


# ### Add covariates for train/normal samples
# 
# This is a bit nuanced since we don't have mutation calling information for the normal samples, so we can't generate a log(mutation burden) covariate.
# 
# For now we'll just take the mean mutation burden from the tumor dataset and apply it to all the normal samples.

# In[11]:


print(mut_burden_df.shape)
mut_burden_df.head()


# In[12]:


# construct covariate matrix for train samples
train_info_df = (
    mut_burden_df.loc[train_samples, :]
      .merge(sample_info_df, left_index=True, right_index=True)    
      .drop(columns={'id_for_stratification'})
      .rename(columns={'cancer_type': 'DISEASE'})
)
print(train_info_df.shape)
train_info_df.head()


# In[13]:


mean_mutation_burden = mut_burden_df.sum() / mut_burden_df.shape[0]
print(mean_mutation_burden)


# In[14]:


# construct covariate matrix for normal samples
normal_info_df = pd.DataFrame(
    {'log10_mut': mean_mutation_burden.values[0]},
    index=normal_ids
)
# add cancer type info for normal samples
normal_info_df = (normal_info_df
    .merge(sample_info_df, left_index=True, right_index=True)    
    .drop(columns={'id_for_stratification'})
    .rename(columns={'cancer_type': 'DISEASE'})
)
print(normal_info_df.shape)
normal_info_df.head()


# In[15]:


def add_dummies_from_model(data_df, info_df, model):
    """TODO: document what info_df looks like, etc"""
    # get cancer type covariates used in original model,
    # in the correct order
    cov_matrix = np.zeros((info_df.shape[0], len(train_cancer_types)))
    for sample_ix, (_, row) in enumerate(info_df.iterrows()):
        try:
            row_cancer_type = row.DISEASE
            cov_ix = train_cancer_types.index(row_cancer_type)
            cov_matrix[sample_ix, cov_ix] = 1
        except ValueError:
            # if cancer type is not in train set (e.g. for a normal sample),
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


# In[16]:


X_train_df = add_dummies_from_model(train_data_df,
                                    train_info_df,
                                    model_fit)


# In[17]:


X_normal_df = add_dummies_from_model(normal_data_df,
                                     normal_info_df,
                                     model_fit)
print(X_normal_df.shape)
X_normal_df.iloc[:5, -20:]


# ### Preprocessing for train/normal samples
# 
# We want to do the following for preprocessing:
# 
# * Make sure we're using the same set of gene features that the model was trained on (or subset to those gene features if not)
# * Standardize the train/normal samples (we'll do this for each dataset independently, for now)

# In[18]:


non_gene_features = list(valid_cancer_types) + ['log10_mut']
train_gene_features = [
    f for f in model_fit.feature_names_in_ if f not in non_gene_features
]
print(train_gene_features[:5])
print(train_gene_features[-5:])


# In[19]:


# we don't want to standardize the non-gene features
is_gene_feature = np.array(
    [(f in train_gene_features) for f in model_fit.feature_names_in_]
)
print(is_gene_feature[:5])
print(is_gene_feature[-5:])
print(sum(is_gene_feature))


# In[20]:


X_train_std_df = tu.standardize_features(X_train_df, is_gene_feature)
X_train_std_df.iloc[:5, :5]


# In[21]:


X_normal_std_df = tu.standardize_features(X_normal_df, is_gene_feature)
X_normal_std_df.iloc[:5, :5]


# ### Make predictions and visualize results

# In[22]:


y_train_preds = model_fit.predict_proba(X_train_std_df)[:, 1]
y_normal_preds = model_fit.predict_proba(X_normal_std_df)[:, 1]

print(y_train_preds.shape)
print(y_normal_preds.shape)


# In[23]:


plot_df = pd.DataFrame(
    {'pred': np.concatenate((y_train_preds, y_normal_preds)),
     'dataset': (['tumor'] * y_train_preds.shape[0]) + (['normal'] * y_normal_preds.shape[0])}
)
plot_df.head()


# In[24]:


sns.set({'figure.figsize': (8, 6)})

sns.violinplot(data=plot_df, x='dataset', y='pred', cut=0)
plt.title('Tumor vs. normal predictions, {}'.format(gene))
plt.xlabel('')
plt.ylabel('Predicted mutation probability')

