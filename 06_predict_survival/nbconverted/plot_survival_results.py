#!/usr/bin/env python
# coding: utf-8

# ## Plot survival prediction results

# In this notebook, we'll compare the results of survival prediction using [elastic net Cox regression](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) for expression and methylation data only.
# 
# The files analyzed in this notebook are generated by the `run_survival_prediction.py` script.
# 
# Notebook parameters:
# * SIG_ALPHA (float): significance cutoff for pairwise comparisons (after FDR correction)

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au
import mpmp.utilities.survival_utilities as su
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# significance cutoff, after FDR correction
SIG_ALPHA = 0.05

# if True, save figures to images directory
SAVE_FIGS = True


# In[3]:


# set results directory
me_results_dir = Path(cfg.repo_root,
                      '06_predict_survival',
                      'results_extended_alphas',
                      'results_1000_pca').resolve()
me_results_desc = 'top 1000 PCs'

# set images directory
images_dir = Path(cfg.images_dirs['survival'])

# load results into a single dataframe
me_results_df = su.load_survival_results(me_results_dir)
me_results_df.rename(columns={'identifier': 'cancer_type',
                              'fold_no': 'fold'}, inplace=True)
me_results_df.head()


# ### Check model convergence results
# 
# In the past we were having issues with model convergence for some cancer types. Let's see how frequently (if at all) this is happening.

# In[4]:


me_count_df = (me_results_df[me_results_df.data_type == 'test']
    .groupby(['cancer_type', 'training_data'])
    .count()
)
problem_df = me_count_df[me_count_df['cindex'] != 16].copy()
print(len(problem_df), '/', len(me_count_df))
problem_df


# We'll just drop these missing cancer types from our comparisons for now, although we could debug the issues with model convergence sometime in the future (e.g. by using an extended parameter range).

# In[5]:


drop_cancer_types = problem_df.index.get_level_values(0).unique().values
print(drop_cancer_types)


# ### Plot survival prediction results
# 
# We want to compare survival prediction for:
# * true labels vs. shuffled labels
# * between omics types
#     
# As a metric, for now we're just using the [censored concordance index](https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.metrics.concordance_index_censored.html). Essentially, this compares the actual order of events (i.e. death or tumor progression) in the test dataset vs. the order of events predicted by the model in the test samples. A higher concordance index = better prediction.

# In[6]:


sns.set({'figure.figsize': (18, 15)})
fig, axarr = plt.subplots(3, 1)

for ix, data_type in enumerate(
    me_results_df.training_data.sort_values().unique()
):
    
    ax = axarr[ix]
    
    filtered_df = me_results_df[
        (me_results_df.training_data == data_type) &
        (me_results_df.data_type == 'test')
    ].copy()

    filtered_df.sort_values(by='cancer_type', inplace=True)

    sns.boxplot(data=filtered_df, x='cancer_type', y='cindex', hue='signal',
                hue_order=['signal', 'shuffled'], ax=ax)
    ax.set_xlabel('TCGA cancer type')
    ax.set_ylabel('Concordance index')
    ax.set_title('Survival prediction using {}, {}'.format(data_type,
                                                           me_results_desc))
    ax.set_ylim(0.0, 1.0)
    
plt.tight_layout()


# In[7]:


me_results_df['identifier'] = (me_results_df.cancer_type + '_' +
                               me_results_df.training_data)
me_results_df.head()


# In[8]:


me_compare_df = au.compare_control_ind(me_results_df,
                                       identifier='identifier',
                                       metric='cindex',
                                       verbose=True)
me_compare_df['cancer_type'] = me_compare_df.identifier.str.split('_', 1, expand=True)[0]
me_compare_df['training_data'] = me_compare_df.identifier.str.split('_', 1, expand=True)[1]

print(len(me_compare_df))
me_compare_df.head()


# In[9]:


sns.set({'figure.figsize': (18, 8)})
    
me_compare_df.sort_values(by='cancer_type', inplace=True)

sns.boxplot(data=me_compare_df[~me_compare_df.cancer_type.isin(drop_cancer_types)],
            x='cancer_type',
            y='delta_cindex',
            hue='training_data',
            hue_order=sorted(me_compare_df.training_data.unique()))
plt.xlabel('TCGA cancer type')
plt.ylabel('cindex(signal) - cindex(shuffled)')
plt.title('Survival prediction, {}'.format(me_results_desc))
plt.ylim(-0.5, 0.5)
    
plt.tight_layout()


# ### Heatmap
# 
# This is similar to the heatmaps we plotted in the results script in `02_classify_mutations` for the mutation prediction problem. We want to compare data types for predicting survival in different cancer types.

# In[10]:


me_all_results_df = au.compare_all_data_types(me_results_df[~me_results_df.cancer_type.isin(drop_cancer_types)],
                                           SIG_ALPHA,
                                           identifier='cancer_type',
                                           metric='cindex')

me_all_results_df.rename(columns={'gene': 'cancer_type'}, inplace=True)
me_all_results_df.sort_values(by='p_value').head(10)


# In[11]:


me_heatmap_df = (me_all_results_df
    .pivot(index='training_data', columns='cancer_type', values='delta_mean')
    .reindex(sorted(me_compare_df.training_data.unique()))
)
me_heatmap_df.iloc[:, :5]


# In[12]:


raw_results_df = (me_results_df
    .drop(columns=['identifier'])
    .rename(columns={'cancer_type': 'identifier'})
)
raw_results_df.head()


# In[13]:


sns.set({'figure.figsize': (28, 6)})
sns.set_context('notebook', font_scale=1.5)

ax = plu.plot_heatmap(me_heatmap_df,
                      me_all_results_df.reset_index(drop=True),
                      different_from_best=True,
                      raw_results_df=raw_results_df,
                      metric='cindex',
                      id_name='cancer_type',
                      scale=(-0.1, 0.4),
                      origin_eps_x=0.01,
                      origin_eps_y=0.01,
                      length_x=0.965,
                      length_y=0.975)

plt.title('Performance by cancer type for survival prediction, {}'.format(me_results_desc), pad=15)
if SAVE_FIGS:
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'survival_me_heatmap.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'survival_me_heatmap.png',
                dpi=300, bbox_inches='tight')


# Key to above heatmap:
# 
# * A blue square = significantly better than label-permuted baseline, but significantly worse than best-performing data type
# * A red square =  significantly better than label-permuted baseline, and not significantly different from best-performing data type (i.e. "statistically equivalent to best")
# * No square/box = not significantly better than label-permuted baseline
# 
# So we can see that many of the same cancer types are well-predicted using all data types (KIRP, LGG, pancancer), and the predictors based on different datasets tend to be statistically equivalent in many of those cases.

# In[14]:


sns.set()
sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 3)

# just use shortened data type names
training_data_map = {k: k for k in sorted(me_all_results_df.training_data.unique())}

plu.plot_volcano_baseline(me_all_results_df,
                          axarr,
                          training_data_map,
                          SIG_ALPHA,
                          identifier='cancer_type',
                          metric='cindex',
                          predict_str='Survival prediction',
                          verbose=True,
                          ylim=(0, 7))

if SAVE_FIGS:
    plt.savefig(images_dir / 'me_vs_shuffled_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'me_vs_shuffled_survival.png',
                dpi=300, bbox_inches='tight')


# In[15]:


me_results_df = (me_results_df
    .drop(columns=['identifier'])
    .rename(columns={'cancer_type': 'identifier'})
).copy()

training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
}
me_results_df.training_data.replace(to_replace=training_data_map, inplace=True)
me_results_df.head()


# In[16]:


# compare expression against all other data modalities
# could do all vs. all, but that would give us lots of plots
sns.set({'figure.figsize': (16, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

plu.plot_volcano_comparison(me_results_df,
                            axarr,
                            training_data_map,
                            SIG_ALPHA,
                            metric='cindex',
                            predict_str='Survival prediction',
                            xlim=(-0.6, 0.6),
                            verbose=True)

# if SAVE_FIGS:
#     plt.savefig(images_dir / 'methylation_comparison.svg', bbox_inches='tight')
#     plt.savefig(images_dir / 'methylation_comparison.png',
#                 dpi=300, bbox_inches='tight')


# ### Same plots, for all data types

# In[17]:


# set results directory
all_results_dir = Path(cfg.repo_root,
                       '06_predict_survival',
                       'results_all_extended_alphas',
                       'results_1000_pca').resolve()
all_results_desc = 'top 1000 PCs'

# load results into a single dataframe
all_data_results_df = su.load_survival_results(all_results_dir)
all_data_results_df.rename(columns={'identifier': 'cancer_type',
                                    'fold_no': 'fold'}, inplace=True)
all_data_results_df.head()


# ### Check model convergence results, all data types

# In[18]:


all_data_count_df = (all_data_results_df[all_data_results_df.data_type == 'test']
    .groupby(['cancer_type', 'training_data'])
    .count()
)
problem_df = all_data_count_df[all_data_count_df['cindex'] != 16].copy()
print(len(problem_df), '/', len(all_data_count_df))
problem_df


# In[19]:


drop_cancer_types = problem_df.index.get_level_values(0).unique().values
print(drop_cancer_types)


# ### Plot raw survival prediction results, all data types

# In[20]:


sns.set()
sns.set({'figure.figsize': (20, 24)})
fig, axarr = plt.subplots(6, 1)

for ix, data_type in enumerate(
    all_data_results_df.training_data.sort_values().unique()
):
    
    ax = axarr[ix]
    
    filtered_df = all_data_results_df[
        (all_data_results_df.training_data == data_type) &
        (all_data_results_df.data_type == 'test')
    ].copy()

    filtered_df.sort_values(by='cancer_type', inplace=True)

    sns.boxplot(data=filtered_df, x='cancer_type', y='cindex', hue='signal',
                hue_order=['signal', 'shuffled'], ax=ax)
    ax.set_xlabel('TCGA cancer type')
    ax.set_ylabel('Concordance index')
    ax.set_title('Survival prediction using {}, {}'.format(data_type,
                                                           all_results_desc))
    ax.set_ylim(0.0, 1.0)
    
plt.tight_layout()


# In[21]:


all_data_results_df['identifier'] = (all_data_results_df.cancer_type + '_' +
                               all_data_results_df.training_data)
all_data_results_df.head()


# In[22]:


all_data_compare_df = au.compare_control_ind(all_data_results_df,
                                       identifier='identifier',
                                       metric='cindex',
                                       verbose=True)
all_data_compare_df['cancer_type'] = all_data_compare_df.identifier.str.split('_', 1, expand=True)[0]
all_data_compare_df['training_data'] = all_data_compare_df.identifier.str.split('_', 1, expand=True)[1]

print(len(all_data_compare_df))
all_data_compare_df.head()


# In[23]:


sns.set({'figure.figsize': (18, 8)})
    
all_data_compare_df.sort_values(by='cancer_type', inplace=True)

sns.boxplot(data=all_data_compare_df[~all_data_compare_df.cancer_type.isin(drop_cancer_types)],
            x='cancer_type',
            y='delta_cindex',
            hue='training_data',
            hue_order=sorted(all_data_compare_df.training_data.unique()))
plt.xlabel('TCGA cancer type')
plt.ylabel('cindex(signal) - cindex(shuffled)')
plt.title('Survival prediction, {}'.format(all_results_desc))
plt.ylim(-0.5, 0.5)
    
plt.tight_layout()


# ### Heatmap, all data types

# In[24]:


all_data_all_results_df = au.compare_all_data_types(all_data_results_df[~all_data_results_df.cancer_type.isin(drop_cancer_types)],
                                                    SIG_ALPHA,
                                                    identifier='cancer_type',
                                                    metric='cindex')

all_data_all_results_df.rename(columns={'gene': 'cancer_type'}, inplace=True)
all_data_all_results_df.sort_values(by='p_value').head(10)


# In[25]:


all_data_heatmap_df = (all_data_all_results_df
    .pivot(index='training_data', columns='cancer_type', values='delta_mean')
    .reindex(sorted(all_data_compare_df.training_data.unique()))
)
all_data_heatmap_df.iloc[:, :5]


# In[26]:


raw_results_df = (all_data_results_df
    .drop(columns=['identifier'])
    .rename(columns={'cancer_type': 'identifier'})
)
raw_results_df.head()


# In[27]:


sns.set({'figure.figsize': (28, 6)})
sns.set_context('notebook', font_scale=1.5)

ax = plu.plot_heatmap(all_data_heatmap_df,
                      all_data_all_results_df.reset_index(drop=True),
                      different_from_best=True,
                      raw_results_df=raw_results_df,
                      metric='cindex',
                      id_name='cancer_type',
                      scale=(-0.1, 0.4),
                      origin_eps_x=0.01,
                      origin_eps_y=0.01,
                      length_x=0.965,
                      length_y=0.95)

plt.title('Performance by cancer type for survival prediction, {}'.format(all_results_desc), pad=15)
if SAVE_FIGS:
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'survival_all_heatmap.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'survival_all_heatmap.png',
                dpi=300, bbox_inches='tight')


# Key to above heatmap:
# 
# * A blue square = significantly better than label-permuted baseline, but significantly worse than best-performing data type
# * A red square =  significantly better than label-permuted baseline, and not significantly different from best-performing data type (i.e. "statistically equivalent to best")
# * No square/box = not significantly better than label-permuted baseline

# In[28]:


sns.set()
sns.set({'figure.figsize': (18, 10)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(2, 3)

# just use shortened data type names
training_data_map = {k: k for k in sorted(all_data_all_results_df.training_data.unique())}

plu.plot_volcano_baseline(all_data_all_results_df,
                          axarr,
                          training_data_map,
                          SIG_ALPHA,
                          identifier='cancer_type',
                          metric='cindex',
                          predict_str='Survival prediction',
                          verbose=True,
                          ylim=(0, 6))

plt.tight_layout()

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_vs_shuffled_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_vs_shuffled_survival.png',
                dpi=300, bbox_inches='tight')


# In[29]:


all_data_results_df = (all_data_results_df
    .drop(columns=['identifier'])
    .rename(columns={'cancer_type': 'identifier'})
).copy()

training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}
all_data_results_df.training_data.replace(to_replace=training_data_map, inplace=True)
all_data_results_df.head()


# In[30]:


# compare expression against all other data modalities
# could do all vs. all, but that would give us lots of plots
sns.set({'figure.figsize': (24, 16)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(2, 3)

plu.plot_volcano_comparison(all_data_results_df,
                            axarr,
                            training_data_map,
                            SIG_ALPHA,
                            metric='cindex',
                            predict_str='Survival prediction',
                            xlim=(-0.6, 0.6),
                            verbose=True)

# if SAVE_FIGS:
#     plt.savefig(images_dir / 'methylation_comparison.svg', bbox_inches='tight')
#     plt.savefig(images_dir / 'methylation_comparison.png',
#                 dpi=300, bbox_inches='tight')

