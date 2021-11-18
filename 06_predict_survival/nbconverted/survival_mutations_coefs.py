#!/usr/bin/env python
# coding: utf-8

# ## Survival mutations: coefficients/predictions analysis
# 
# In this notebook, we want to look a bit closer at what features are being selected in the predictions -> survival analysis, to get an idea of which genes are used by the model and what their predictions look like.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# set results directories
results_dir = Path(cfg.results_dirs['survival'], 'mutations_me_all')

# set list of PCA component numbers to look for
n_pcs = 10


# In[3]:


# order to plot data types in
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'vogelstein_mutations': 'all Vogelstein mutations',
    'significant_mutations': 'significant mutations',
    'mutation_preds_expression': 'mutation scores, expression',
    'mutation_preds_me_27k': 'mutation scores, 27k',
    'mutation_preds_me_450k': 'mutation scores, 450k',
}


# ### Pan-cancer models

# In[4]:


def is_feature_covariate(features):
    sample_info_df = du.load_sample_info('expression')
    cancer_types = list(sample_info_df.cancer_type.unique())
    covariate_names = cancer_types + ['age', 'log10_mut']
    return [(f in covariate_names) for f in features]


# In[5]:


cancer_type = 'pancancer'
training_data = 'mutation_preds_me_450k'
seed = 42
fold = 0

coefs_file = '{}_{}_signal_survival_s{}_coefficients.tsv.gz'.format(
    cancer_type, training_data, seed)
coefs_file = results_dir / coefs_file

coefs_df = pd.read_csv(coefs_file, sep='\t')
coefs_df = coefs_df.loc[coefs_df.fold == fold, :].copy()
coefs_df['is_covariate'] = is_feature_covariate(coefs_df.feature)
coefs_df.head()


# In[6]:


sns.set({'figure.figsize': (20, 5)})

plot_df = (coefs_df
    .sort_values(by='abs', ascending=False)
    .iloc[:50, :]
    .reset_index()
)

sns.barplot(data=plot_df, x=plot_df.index, y=plot_df.weight,
            dodge=False, hue='is_covariate', hue_order=[True, False])
plt.title('Top coefficients for {}, cancer type {}'.format(training_data, cancer_type))
plt.gca().set_xticklabels(plot_df.feature)
for tick in plt.gca().get_xticklabels():
    tick.set_rotation(45)


# In[7]:


data_type = training_data.replace('mutation_preds_', '')

predictions_file = cfg.predictions[data_type]
preds_df = pd.read_csv(predictions_file, sep='\t', index_col=0)
preds_df.index.rename('sample_id', inplace=True)
print(preds_df.shape)
preds_df.head()


# In[8]:


pancan_data = du.load_pancancer_data(verbose=True)
mutation_df = pancan_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# In[11]:


# gene = 'IDH1'
gene = 'CARD11'
plot_df = preds_df[[gene]].merge(mutation_df[[gene]],
                                 left_index=True, right_index=True)
plot_df.rename(columns={'{}_x'.format(gene): 'pred_scores',
                        '{}_y'.format(gene): 'is_mutated'},
               inplace=True)
plot_df['pred_prob'] = 1 / (1 + np.exp(-plot_df.pred_scores))
print(plot_df.shape)
print(plot_df.pred_prob.min())
print(plot_df.pred_prob.max())
plot_df.head()


# In[12]:


sns.set({'figure.figsize': (10, 6)})
sns.violinplot(data=plot_df, x='is_mutated', y='pred_prob')
plt.title(gene)
plt.xlabel('True mutation status')
plt.ylabel('Predicted mutation probability')

