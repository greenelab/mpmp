#!/usr/bin/env python
# coding: utf-8

# ## Compare multi-omics results for MLP and elastic net
# 
# In this analysis we want to directly compare performance on the 6 gene multi-omics pilot experiment for our 3-layer MLP neural networks with our original elastic net logistic regression models.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


grid_results_dir = Path(
    cfg.results_dirs['multimodal'],
    'compressed_shuffle_cancer_type',
    'gene'
).resolve()

grid_unimodal_results_dir = Path(
    cfg.results_dirs['mutation'],
    'methylation_results_shuffle_cancer_type',
    'gene'
)

bayes_opt_results_dir = Path(
    cfg.results_dirs['multimodal'],
    'bayes_opt',
    'gene'
).resolve()

mlp_results_dir = Path(
    cfg.results_dirs['multimodal'],
    'mlp_pilot',
    'gene'
).resolve()


# In[3]:


# load raw data
grid_results_df = au.load_stratified_prediction_results(grid_results_dir, 'gene')

# drop TET2 for now
grid_results_df = grid_results_df[~(grid_results_df.identifier.isin(['TET2']))]
grid_results_df['model'] = 'elasticnet, grid'

# make sure that we have data for all data types and for two replicates (random seeds)
print(grid_results_df.shape)
print(grid_results_df.seed.unique())
print(grid_results_df.identifier.unique())
print(grid_results_df.training_data.unique())
grid_results_df.head()


# In[4]:


# load expression and me_27k results
u_results_df = au.load_compressed_prediction_results(grid_unimodal_results_dir, 'gene')
genes = grid_results_df.identifier.unique()
u_results_df = u_results_df[(u_results_df.n_dims == 5000) &
                            (u_results_df.identifier.isin(genes))].copy()
u_results_df.drop(columns='n_dims', inplace=True)
u_results_df['model'] = 'elasticnet, grid'

# make sure data loaded matches our expectations
print(u_results_df.shape)
print(u_results_df.seed.unique())
print(u_results_df.identifier.unique())
print(u_results_df.training_data.unique())
u_results_df.head()


# In[5]:


grid_results_df = pd.concat((
    grid_results_df, u_results_df
))
group_df = (grid_results_df
    .groupby(['identifier', 'training_data'])
    .count()
)

# all of these should come up as duplicates (so only one row will print)
# if not, there are either missing folds or duplicated folds
group_df[~group_df.duplicated()].head(10)


# In[6]:


# load raw data
bayes_opt_results_df = au.load_stratified_prediction_results(
    bayes_opt_results_dir, 'gene')

bayes_opt_results_df = bayes_opt_results_df[~(bayes_opt_results_df.identifier.isin(['TET2']))]
bayes_opt_results_df = bayes_opt_results_df[~(bayes_opt_results_df.training_data.isin(['expression']))].copy()
bayes_opt_results_df['model'] = 'elasticnet, bayes'

# make sure that we have data for all data types and for two replicates (random seeds)
print(bayes_opt_results_df.shape)
print(bayes_opt_results_df.seed.unique())
print(bayes_opt_results_df.identifier.unique())
print(bayes_opt_results_df.training_data.unique())
bayes_opt_results_df.head()


# In[7]:


# load expression and me_27k results
bayes_u_results_df = au.load_compressed_prediction_results(
    bayes_opt_results_dir, 'gene')

# filter to genes/dims we're using here
bayes_u_results_df = bayes_u_results_df[
    (bayes_u_results_df.n_dims == 5000) &
    (bayes_u_results_df.identifier.isin(genes))
].copy()
bayes_u_results_df.drop(columns='n_dims', inplace=True)
bayes_u_results_df['model'] = 'elasticnet, bayes'

# make sure data loaded matches our expectations
print(bayes_u_results_df.shape)
print(bayes_u_results_df.seed.unique())
print(bayes_u_results_df.identifier.unique())
print(bayes_u_results_df.training_data.unique())
bayes_u_results_df.head()


# In[8]:


bayes_opt_results_df = pd.concat((
    bayes_opt_results_df, bayes_u_results_df
))
group_df = (bayes_opt_results_df
    .groupby(['identifier', 'training_data'])
    .count()
)

# all of these should come up as duplicates (so only one row will print)
# if not, there are either missing folds or duplicated folds
group_df[~group_df.duplicated()].head(10)


# In[9]:


# load raw data
mlp_results_df = au.load_compressed_prediction_results(mlp_results_dir, 'gene', multimodal=True)
mlp_results_df = mlp_results_df.loc[
    mlp_results_df.training_data.str.contains('\.'), :
]
mlp_results_df.loc[mlp_results_df.n_dims == 1000, 'model'] = 'mlp, random, 1000 PCs'
mlp_results_df.loc[mlp_results_df.n_dims == 5000, 'model'] = 'mlp, random, 5000 PCs'
mlp_results_df.drop(columns='n_dims', inplace=True)

# make sure that we have data for all data types and for two replicates (random seeds)
print(mlp_results_df.shape)
print(mlp_results_df.seed.unique())
print(mlp_results_df.identifier.unique())
print(mlp_results_df.training_data.unique())
print(mlp_results_df.model.unique())
mlp_results_df.head()


# In[10]:


mlp_u_results_df = au.load_compressed_prediction_results(mlp_results_dir, 'gene')
mlp_u_results_df.loc[mlp_u_results_df.n_dims == 1000, 'model'] = 'mlp, random, 1000 PCs'
mlp_u_results_df.loc[mlp_u_results_df.n_dims == 5000, 'model'] = 'mlp, random, 5000 PCs'
mlp_u_results_df.drop(columns='n_dims', inplace=True)
mlp_u_results_df = mlp_u_results_df.loc[mlp_u_results_df.model == 'mlp, random, 5000 PCs', :]

# make sure data loaded matches our expectations
print(mlp_u_results_df.shape)
print(mlp_u_results_df.seed.unique())
print(mlp_u_results_df.identifier.unique())
print(mlp_u_results_df.training_data.unique())
print(mlp_u_results_df.model.unique())
mlp_u_results_df.head()


# In[11]:


mlp_results_df = pd.concat((
    mlp_results_df, mlp_u_results_df
))
group_df = (mlp_results_df
    .groupby(['identifier', 'training_data'])
    .count()
)

# all of these should come up as duplicates (so only one row will print)
# if not, there are either missing folds or duplicated folds
group_df[~group_df.duplicated()].head(10)


# In[12]:


results_df = pd.concat((
    grid_results_df, bayes_opt_results_df, mlp_results_df
))

print(results_df.shape)
results_df.head()


# In[13]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (20, 14)})
fig, axarr = plt.subplots(2, 3)
results_df.sort_values(by=['identifier', 'signal', 'training_data'], inplace=True)

data_order =['expression',
             'me_27k',
             'me_450k',
             'expression.me_27k',
             'expression.me_450k',
             'me_27k.me_450k',
             'expression.me_27k.me_450k']

for ix, gene in enumerate(results_df.identifier.unique()):
    ax = axarr[ix // 3, ix % 3]
    plot_df = results_df[(results_df.identifier == gene) &
                         (results_df.data_type == 'test') &
                         (results_df.signal == 'signal')]
    sns.boxplot(data=plot_df, x='training_data', y='aupr',
                order=data_order, hue='model', ax=ax)
    ax.set_title('Prediction for {} mutation'.format(gene))
    ax.set_ylabel('AUPR')
    ax.set_ylim(0.0, 1.1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(40)
    if ix != 0:
        ax.legend_.remove()
        
plt.tight_layout()


# In[14]:


# then, for each training data type, get the AUPR difference between signal and shuffled
compare_df = pd.DataFrame()
for training_data in results_df.training_data.unique():
    for model in results_df.model.unique():
        results_df.sort_values(by=['seed', 'fold'], inplace=True)
        data_compare_df = au.compare_control_ind(
            results_df[(results_df.training_data == training_data) &
                       (results_df.model == model)],
            identifier='identifier',
            metric='aupr',
            verbose=True
        )
        data_compare_df['training_data'] = training_data
        data_compare_df['model'] = model
        data_compare_df.rename(columns={'identifier': 'gene'}, inplace=True)
        compare_df = pd.concat((compare_df, data_compare_df))
    
compare_df.head(10)


# In[15]:


# each subplot will show results for one gene
sns.set({'figure.figsize': (20, 17)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(2, 3)

# don't show bayes results for this figure
compare_df = compare_df[~compare_df.model.str.contains('bayes')]
compare_df['model'] = (compare_df.model.str
    .replace('elasticnet, grid', 'elastic net')
    .replace('mlp, random, 5000 PCs', 'fully-connected NN')
)

data_names = {
    'expression': 'gene expression',
    'me_27k': '27K methylation',
    'me_450k': '450K methylation',
    'expression.me_27k': 'expression + 27K methylation',
    'expression.me_450k': 'expression + 450K methylation',
    'me_27k.me_450k': '27K methylation + 450K methylation',
    'expression.me_27k.me_450k': 'expression + 27K methylation + 450K methylation'
}

plu.plot_multi_omics_results(compare_df,
                             axarr,
                             data_names,
                             colors=[],
                             metric='aupr')
plt.tight_layout()

images_dir = Path(cfg.images_dirs['multimodal'], 'mlp')
images_dir.mkdir(exist_ok=True)

svg_filename = 'multi_omics_compare_mlp.svg'
png_filename = 'multi_omics_compare_mlp.png'
plt.savefig(images_dir / svg_filename, bbox_inches='tight')
plt.savefig(images_dir / png_filename, dpi=300, bbox_inches='tight')


# In general we can see that the MLP sometimes outperforms the elastic net models, particularly on the multi-omics data (e.g. for EGFR and TP53), but sometimes it considerably underperforms the elastic net models, particularly on single-omics data (e.g. for PIK3CA and EGFR). Performance for the elastic net models seems to be a bit more robust across cross-validation folds and hyperparameter choices.
