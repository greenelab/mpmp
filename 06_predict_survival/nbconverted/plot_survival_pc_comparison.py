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
import mpmp.utilities.survival_utilities as su
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Results for gene expression vs. DNA methylation comparison

# In[2]:


# significance cutoff, after FDR correction
SIG_ALPHA = 0.05

# if True, save figures to images directory
SAVE_FIGS = True
images_dir = Path(cfg.images_dirs['survival'])

# set results directory
me_results_dir = Path(cfg.repo_root,
                      '06_predict_survival',
                      'results_extended_alphas')

# set list of PCA component numbers to look for
pcs_list = [10, 100, 500, 1000, 5000]

# order to plot data types in
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
}


# In[3]:


me_data_count_df = []

for n_dim in pcs_list:
    # load results into a single dataframe
    me_pcs_dir = Path(me_results_dir, 'results_{}_pca'.format(n_dim))
    me_results_df = su.load_survival_results(me_pcs_dir)
    me_results_df.rename(columns={'identifier': 'cancer_type',
                                  'fold_no': 'fold'}, inplace=True)
    
    me_count_df = (me_results_df[me_results_df.data_type == 'test']
        .groupby(['cancer_type', 'training_data'])
        .count()
    )
    problem_df = me_count_df[me_count_df['cindex'] != 16].copy()
    drop_cancer_types = problem_df.index.get_level_values(0).unique().values
    
    me_all_results_df = au.compare_all_data_types(
        me_results_df[~me_results_df.cancer_type.isin(drop_cancer_types)],
        SIG_ALPHA,
        identifier='cancer_type',
        metric='cindex')
    
    print(n_dim)
    num_reject = me_all_results_df.groupby(['training_data']).sum().reject_null
    num_total = me_all_results_df.groupby(['training_data']).count().reject_null
    
    dim_count_df = (
            pd.DataFrame(num_reject).rename(columns={'reject_null': 'num_reject'})
        .merge(
            pd.DataFrame(num_total).rename(columns={'reject_null': 'num_total'})
        , left_index=True, right_index=True)
    )
    dim_count_df['n_dim'] = n_dim
    dim_count_df['ratio'] = dim_count_df.num_reject / dim_count_df.num_total
    me_data_count_df.append(dim_count_df)
    
me_data_count_df = pd.concat(me_data_count_df).reset_index()
me_data_count_df.training_data.replace(to_replace=training_data_map, inplace=True)
me_data_count_df.head(10)


# In[4]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(1, 3)

sns.pointplot(data=me_data_count_df, x='n_dim', y='num_reject',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[0])
axarr[0].set_xlabel('Number of PCs')
axarr[0].set_ylabel('Number of well-predicted cancer types')
axarr[0].set_title('Well-predicted cancer type count vs. PC count')

sns.pointplot(data=me_data_count_df, x='n_dim', y='num_total',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[1])
axarr[1].set_xlabel('Number of PCs')
axarr[1].set_ylabel('Total valid cancer types')
axarr[1].set_title('Valid cancer type count vs. PC count')

sns.pointplot(data=me_data_count_df, x='n_dim', y='ratio',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[2])
axarr[2].set_xlabel('Number of PCs')
axarr[2].set_ylabel('Ratio of well-predicted to total cancer type')
axarr[2].set_title('Well-predicted/total ratio vs. PC count')


# In[5]:


me_performance_df = []
all_drop_cancer_types = set()

for n_dim in pcs_list:
    # load results into a single dataframe
    me_pcs_dir = Path(me_results_dir, 'results_{}_pca'.format(n_dim))
    me_results_df = su.load_survival_results(me_pcs_dir)
    me_results_df.rename(columns={'identifier': 'cancer_type',
                                  'fold_no': 'fold'}, inplace=True)
    
    me_count_df = (me_results_df[me_results_df.data_type == 'test']
        .groupby(['cancer_type', 'training_data'])
        .count()
    )
    problem_df = me_count_df[me_count_df['cindex'] != 16].copy()
    drop_cancer_types = problem_df.index.get_level_values(0).unique().values
    
    me_all_results_df = au.compare_all_data_types(
        me_results_df[~me_results_df.cancer_type.isin(drop_cancer_types)],
        SIG_ALPHA,
        filter_genes=False,
        compare_ind=True,
        identifier='cancer_type',
        metric='cindex')
    me_all_results_df.rename(columns={'gene': 'cancer_type'}, inplace=True)
    me_all_results_df['n_dim'] = n_dim
    me_performance_df.append(me_all_results_df)
    
me_performance_df = pd.concat(me_performance_df)
me_performance_df.training_data.replace(to_replace=training_data_map, inplace=True)
me_performance_df.head(10)


# In[6]:


group_cancer_types = me_performance_df.groupby(['cancer_type']).count().n_dim
max_count = group_cancer_types.max()
valid_cancer_types = group_cancer_types[group_cancer_types == max_count].index
print(valid_cancer_types)


# In[7]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.pointplot(data=me_performance_df, x='n_dim', y='delta_cindex', hue='training_data',
              hue_order=training_data_map.values())
plt.xlabel('Number of PCs')
plt.ylabel('cindex(signal) - cindex(shuffled)')
plt.title('Performance for varying PC count, averaged over cancer types')


# In[8]:


cancer_type_avg = (
    me_performance_df[me_performance_df.cancer_type.isin(valid_cancer_types)]
      .groupby('cancer_type')
      .mean()
).delta_cindex
cancer_type_avg.sort_values(ascending=False).head(10)


# In[9]:


cancer_type_sd = me_performance_df.groupby('cancer_type').std().delta_cindex
cancer_type_cv = cancer_type_avg / cancer_type_sd
cancer_type_cv.sort_values(ascending=False).head(10)


# In[10]:


sns.set({'figure.figsize': (7, 6)})
sns.set_style('whitegrid')

sns.pointplot(data=me_performance_df[me_performance_df.cancer_type == 'pancancer'],
              x='n_dim', y='delta_cindex', hue='training_data', 
              hue_order=training_data_map.values())
plt.xlabel('Number of PCs', size=14)
plt.ylabel('cindex(signal) - cindex(shuffled)', size=14)
plt.legend(title='Training data', fontsize=13, title_fontsize=13)
plt.title('Pan-cancer survival performance, expression/methylation', size=14)
plt.ylim(-0.05, 0.25)

if SAVE_FIGS:
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'me_pancan_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'me_pancan_survival.png',
                dpi=300, bbox_inches='tight')


# In[11]:


sns.set({'figure.figsize': (28, 5)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(1, 5)

cancer_type_cv = cancer_type_cv[cancer_type_cv.index != 'pancancer']
for ix, cancer_type in enumerate(cancer_type_cv.sort_values(ascending=False).index[:5]):
    ax = axarr[ix]
    sns.pointplot(data=me_performance_df[me_performance_df.cancer_type == cancer_type],
                  x='n_dim', y='delta_cindex', hue='training_data',
                  hue_order=training_data_map.values(), ax=ax)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('cindex(signal) - cindex(shuffled)')
    ax.set_title('{} survival performance'.format(cancer_type))
    ax.set_ylim(-0.05, 0.4)
    if ix != 0:
        ax.get_legend().remove()
        
if SAVE_FIGS:
    plt.savefig(images_dir / 'me_top_cancers_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'me_top_cancers_survival.png',
                dpi=300, bbox_inches='tight')


# ### Results for all data types comparison

# In[12]:


# set results directory
all_results_dir = Path(cfg.repo_root,
                      '06_predict_survival',
                      'results_all_extended_alphas')

# order to plot data types in
training_data_map = {
    'expression': 'gene expression',
    'me_27k': '27k methylation',
    'me_450k': '450k methylation',
    'rppa': 'RPPA',
    'mirna': 'microRNA',
    'mut_sigs': 'mutational signatures',
}


# In[13]:


all_data_count_df = []

for n_dim in pcs_list:
    # load results into a single dataframe
    all_pcs_dir = Path(all_results_dir, 'results_{}_pca'.format(n_dim))
    all_results_df = su.load_survival_results(all_pcs_dir)
    all_results_df.rename(columns={'identifier': 'cancer_type',
                                  'fold_no': 'fold'}, inplace=True)
    
    all_count_df = (all_results_df[all_results_df.data_type == 'test']
        .groupby(['cancer_type', 'training_data'])
        .count()
    )
    problem_df = all_count_df[all_count_df['cindex'] != 16].copy()
    drop_cancer_types = problem_df.index.get_level_values(0).unique().values
    
    all_all_results_df = au.compare_all_data_types(
        all_results_df[~all_results_df.cancer_type.isin(drop_cancer_types)],
        SIG_ALPHA,
        identifier='cancer_type',
        metric='cindex')
    
    print(n_dim)
    num_reject = all_all_results_df.groupby(['training_data']).sum().reject_null
    num_total = all_all_results_df.groupby(['training_data']).count().reject_null
    
    dim_count_df = (
            pd.DataFrame(num_reject).rename(columns={'reject_null': 'num_reject'})
        .merge(
            pd.DataFrame(num_total).rename(columns={'reject_null': 'num_total'})
        , left_index=True, right_index=True)
    )
    dim_count_df['n_dim'] = n_dim
    dim_count_df['ratio'] = dim_count_df.num_reject / dim_count_df.num_total
    all_data_count_df.append(dim_count_df)
    
all_data_count_df = pd.concat(all_data_count_df).reset_index()
all_data_count_df.training_data.replace(to_replace=training_data_map, inplace=True)
all_data_count_df.head(10)


# In[14]:


sns.set({'figure.figsize': (22, 5)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(1, 3)

sns.pointplot(data=all_data_count_df, x='n_dim', y='num_reject',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[0])
axarr[0].set_xlabel('Number of PCs')
axarr[0].set_ylabel('Number of well-predicted cancer types')
axarr[0].set_title('Well-predicted cancer type count vs. PC count')

sns.pointplot(data=all_data_count_df, x='n_dim', y='num_total',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[1])
axarr[1].set_xlabel('Number of PCs')
axarr[1].set_ylabel('Total valid cancer types')
axarr[1].set_title('Valid cancer type count vs. PC count')

sns.pointplot(data=all_data_count_df, x='n_dim', y='ratio',
              hue='training_data', hue_order=training_data_map.values(), ax=axarr[2])
axarr[2].set_xlabel('Number of PCs')
axarr[2].set_ylabel('Ratio of well-predicted to total cancer type')
axarr[2].set_title('Well-predicted/total ratio vs. PC count')


# In[15]:


all_performance_df = []
all_drop_cancer_types = set()

for n_dim in pcs_list:
    # load results into a single dataframe
    all_pcs_dir = Path(all_results_dir, 'results_{}_pca'.format(n_dim))
    all_results_df = su.load_survival_results(all_pcs_dir)
    all_results_df.rename(columns={'identifier': 'cancer_type',
                                  'fold_no': 'fold'}, inplace=True)
    
    all_count_df = (all_results_df[all_results_df.data_type == 'test']
        .groupby(['cancer_type', 'training_data'])
        .count()
    )
    problem_df = all_count_df[all_count_df['cindex'] != 16].copy()
    drop_cancer_types = problem_df.index.get_level_values(0).unique().values
    
    all_all_results_df = au.compare_all_data_types(
        all_results_df[~all_results_df.cancer_type.isin(drop_cancer_types)],
        SIG_ALPHA,
        filter_genes=False,
        compare_ind=True,
        identifier='cancer_type',
        metric='cindex')
    all_all_results_df.rename(columns={'gene': 'cancer_type'}, inplace=True)
    all_all_results_df['n_dim'] = n_dim
    all_performance_df.append(all_all_results_df)
    
all_performance_df = pd.concat(all_performance_df)
all_performance_df.training_data.replace(to_replace=training_data_map, inplace=True)
all_performance_df.head(10)


# In[16]:


group_cancer_types = all_performance_df.groupby(['cancer_type']).count().n_dim
max_count = group_cancer_types.max()
valid_cancer_types = group_cancer_types[group_cancer_types == max_count].index
print(valid_cancer_types)


# In[17]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

sns.pointplot(data=all_performance_df, x='n_dim', y='delta_cindex', hue='training_data',
              hue_order=training_data_map.values())
plt.xlabel('Number of PCs')
plt.ylabel('cindex(signal) - cindex(shuffled)')
plt.title('Performance for varying PC count, averaged over cancer types')


# In[18]:


cancer_type_avg = (
    all_performance_df[all_performance_df.cancer_type.isin(valid_cancer_types)]
      .groupby('cancer_type')
      .mean()
).delta_cindex
cancer_type_avg.sort_values(ascending=False).head(10)


# In[19]:


cancer_type_sd = all_performance_df.groupby('cancer_type').std().delta_cindex
cancer_type_cv = cancer_type_avg / cancer_type_sd
cancer_type_cv.sort_values(ascending=False).head(10)


# In[20]:


sns.set({'figure.figsize': (10, 5)})
sns.set_style('whitegrid')

sns.pointplot(data=all_performance_df[all_performance_df.cancer_type == 'pancancer'],
              x='n_dim', y='delta_cindex', hue='training_data', 
              hue_order=training_data_map.values())
plt.xlabel('Number of PCs', size=14)
plt.ylabel('cindex(signal) - cindex(shuffled)', size=14)
plt.legend(title='Training data', fontsize=13, title_fontsize=13, loc='upper left', ncol=2)
plt.title('Pan-cancer survival performance, all data types', size=14)
plt.ylim(-0.05, 0.3)

if SAVE_FIGS:
    plt.savefig(images_dir / 'all_pancan_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_pancan_survival.png',
                dpi=300, bbox_inches='tight')


# In[21]:


sns.set({'figure.figsize': (28, 5)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(1, 5)

cancer_type_cv = cancer_type_cv[cancer_type_cv.index != 'pancancer']
for ix, cancer_type in enumerate(cancer_type_cv.sort_values(ascending=False).index[:5]):
    ax = axarr[ix]
    sns.pointplot(data=all_performance_df[all_performance_df.cancer_type == cancer_type],
                  x='n_dim', y='delta_cindex', hue='training_data',
                  hue_order=training_data_map.values(), ax=ax)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('cindex(signal) - cindex(shuffled)')
    ax.set_title('{} survival performance'.format(cancer_type))
    ax.set_ylim(-0.05, 0.4)
    if ix != 0:
        ax.get_legend().remove()
        
if SAVE_FIGS:
    plt.savefig(images_dir / 'all_top_cancers_survival.svg', bbox_inches='tight')
    plt.savefig(images_dir / 'all_top_cancers_survival.png',
                dpi=300, bbox_inches='tight')

