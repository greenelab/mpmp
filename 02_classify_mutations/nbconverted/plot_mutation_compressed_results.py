#!/usr/bin/env python
# coding: utf-8

# ## Plot compressed mutation prediction results
# 
# Here, we'll look at the results of our mutation prediction experiments with compressed (i.e. PCA projected) data. The idea is similar to `01_classify_stratified/plot_mutation_results.ipynb`, but with slightly more parameters (e.g. number of PCA components).
# 
# Notebook parameters:
# * SIG_ALPHA (float): significance cutoff (after FDR correction)

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

import mpmp.config as cfg
import mpmp.utilities.analysis_utilities as au


# In[2]:


# set results directory
results_dir = Path(cfg.results_dirs['mutation'],
                   'compressed_results', 'gene').resolve()
# set significance cutoff after FDR correction
SIG_ALPHA = 0.001


# In[3]:


# load raw data
results_df = au.load_compressed_prediction_results(results_dir,
                                                   'gene',
                                                   old_filenames=True)
print(results_df.shape)
results_df.head()


# In[4]:


# get dataframes comparing each data type to their shuffled baseline
compare_df = pd.DataFrame()
for n_dims in results_df.n_dims.unique():
    for training_data_type in results_df.training_data.unique():
        data_df = results_df[
            (results_df.training_data == training_data_type) &
            (results_df.n_dims == n_dims)
        ]
        # note that we're doing FDR correction separately for each n_dims
        # and training data type
        data_results_df = au.compare_results(data_df,
                                             identifier='identifier',
                                             metric='aupr',
                                             correction=True,
                                             correction_method='fdr_bh',
                                             correction_alpha=SIG_ALPHA,
                                             verbose=False)
        data_results_df['training_data'] = training_data_type
        data_results_df['n_dims'] = n_dims
        compare_df = pd.concat((compare_df, data_results_df))
compare_df.head()


# In[5]:


def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        if point['y'] > -np.log10(SIG_ALPHA):
            text_labels.append(
                ax.text(point['x'], point['y'], str(point['gene']))
            )
    return text_labels

compare_df = (
    compare_df.rename(columns={'identifier': 'gene'})
              .sort_values(by=['n_dims', 'training_data'])
)
compare_df['nlog10_p'] = -np.log10(compare_df.corr_pval)

sns.set({'figure.figsize': (20, 16)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(3, 3)

# plot mutation prediction from various data types, in a volcano-like plot
for row_ix, n_dims in enumerate(compare_df.n_dims.unique()):
    for col_ix, training_data_type in enumerate(compare_df.training_data.unique()):
        ax = axarr[row_ix, col_ix]
        data_df = compare_df[
            (compare_df.training_data == training_data_type) &
            (compare_df.n_dims == n_dims)
        ]
        sns.scatterplot(data=data_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                        hue_order=[False, True], ax=ax)
        # add vertical line at 0
        ax.axvline(x=0, linestyle='--', color='black', linewidth=1.25)
        # add horizontal line at statistical significance threshold
        l = ax.axhline(y=-np.log10(SIG_ALPHA), linestyle='--', linewidth=1.25)
        # label horizontal line with significance threshold
        # (matplotlib makes this fairly difficult, sadly)
        ax.text(0.9, -np.log10(SIG_ALPHA)+0.2,
                r'$\mathbf{{\alpha = {}}}$'.format(SIG_ALPHA),
                va='center', ha='center', color=l.get_color(),
                backgroundcolor=ax.get_facecolor())
        ax.set_xlabel('AUPR(signal) - AUPR(shuffled)')
        ax.set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
        ax.set_xlim((-0.2, 1.0))
        y_max = max(compare_df.nlog10_p)
        ax.set_ylim((0, y_max+2))
        ax.legend(title=r'Reject $H_0$', loc='upper left')
        ax.set_title(r'{} PCs, {} data ({}/{} significant genes)'.format(
            n_dims, training_data_type,
            np.count_nonzero(data_df.reject_null),
            data_df.shape[0]
        ), size=13, pad=10)

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand in inkscape if necessary
        text_labels = label_points(data_df['delta_mean'],
                                   data_df['nlog10_p'],
                                   data_df.gene,
                                   data_df.reject_null,
                                   ax)
        adjust_text(text_labels,
                    ax=ax, 
                    expand_text=(1., 1.),
                    lim=5)
        
plt.suptitle('Mutation prediction, signal vs. shuffled results', size=16)
plt.tight_layout(w_pad=2, h_pad=2)
plt.subplots_adjust(top=0.94)


# In[6]:


sns.set({'figure.figsize': (18, 6)})
fig, axarr = plt.subplots(1, 2)

# plot mean performance over all genes in Vogelstein dataset
# sns.pointplot(data=compare_df, x='n_dims', y='delta_mean', hue='training_data', ax=axarr[0])
sns.boxplot(data=compare_df, x='n_dims', y='delta_mean', hue='training_data', ax=axarr[0])
axarr[0].set_title('Mean mutation prediction, performance vs. PCA components')
axarr[0].set_xlabel('Number of PCA components')
axarr[0].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[0].set_ylim(-0.3, max(compare_df.delta_mean + 0.05))

# plot mean performance for genes that are significant for at least one data type
gene_list = compare_df[compare_df.reject_null == True].gene.unique()
sns.boxplot(data=compare_df[compare_df.gene.isin(gene_list)], x='n_dims', y='delta_mean', hue='training_data', ax=axarr[1])
# sns.pointplot(data=compare_df[compare_df.gene.isin(gene_list)], 
#               x='n_dims', y='delta_mean', hue='training_data', ax=axarr[1])
axarr[1].set_title('Mean prediction for sig genes only, performance vs. PCA components')
axarr[1].set_xlabel('Number of PCA components')
axarr[1].set_ylabel('AUPR(signal) - AUPR(shuffled)')
axarr[1].set_ylim(-0.2, max(compare_df.delta_mean + 0.05))


# In[7]:


# get results for full gene expression/27k methylation datasets
# we want to compare compressed to full results below
raw_results_dir = Path(cfg.results_dirs['mutation'], 'mutation_imputed_n10_i5', 'gene').resolve()
raw_results_df = au.load_stratified_prediction_results(raw_results_dir, 'gene')

raw_expression_df = (
    raw_results_df[raw_results_df.training_data == 'expression']
        .drop(columns=['training_data'])
)
raw_expression_results_df = au.compare_results(raw_expression_df,
                                               identifier='identifier',
                                               metric='aupr',
                                               correction=True,
                                               correction_method='fdr_bh',
                                               correction_alpha=SIG_ALPHA,
                                               verbose=False)
raw_expression_results_df.rename(columns={'identifier': 'gene'}, inplace=True)


raw_methylation_df = (
    raw_results_df[raw_results_df.training_data == 'methylation']
        .drop(columns=['training_data'])
)
raw_methylation_results_df = au.compare_results(raw_methylation_df,
                                                identifier='identifier',
                                                metric='aupr',
                                                correction=True,
                                                correction_method='fdr_bh',
                                                correction_alpha=SIG_ALPHA,
                                                verbose=False)
raw_methylation_results_df.rename(columns={'identifier': 'gene'}, inplace=True)


# In[8]:


# line plot of performance over increasing number of PCs
# plot dotted line for expression/me performance with all genes?
gene = 'IDH1'
sns.set({'figure.figsize': (8, 5)})
gene_df = compare_df[compare_df.gene == gene]
sns.pointplot(data=gene_df, x='n_dims', y='delta_mean', hue='training_data')
raw_expression_val = (
    raw_expression_results_df[raw_expression_results_df.gene == gene]
        .delta_mean.values[0]
)
plt.axhline(y=raw_expression_val, linestyle='--', color=sns.color_palette()[0])
raw_me_27k_val = (
    raw_methylation_results_df[raw_methylation_results_df.gene == gene]
        .delta_mean.values[0]
)
plt.axhline(y=raw_me_27k_val, linestyle='--', color=sns.color_palette()[1])
plt.title('{} mutation prediction, performance vs. PCA components'.format(gene))
plt.xlabel('Number of PCA components')
plt.ylabel('AUPR(signal) - AUPR(shuffled)')
plt.ylim(0, max(max(gene_df.delta_mean), raw_expression_val, raw_me_27k_val)+0.05)


# In[9]:


def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        if point['y'] > 1.0:
            text_labels.append(
                ax.text(point['x'], point['y'], str(point['gene']))
            )
    return text_labels

# plot comparisons for all pairwise combinations of training datasets,
# within each choice of compression dimension
import itertools as it

sns.set({'figure.figsize': (20, 16)})
sns.set_style('whitegrid')
fig, axarr = plt.subplots(3, 3)

results_df.sort_values(by=['n_dims', 'training_data'], inplace=True)
for row_ix, n_dims in enumerate(results_df.n_dims.unique()):
    for col_ix, (train_1, train_2) in enumerate(
            it.combinations(results_df.training_data.unique(), 2)):
        train_1_df = results_df[
            (results_df.n_dims == n_dims) &
            (results_df.training_data == train_1)
        ]
        train_2_df = results_df[
            (results_df.n_dims == n_dims) &
            (results_df.training_data == train_2)
        ]
        compare_train_df = au.compare_results(train_1_df,
                                              train_2_df,
                                              identifier='identifier',
                                              metric='aupr',
                                              correction=True,
                                              correction_method='fdr_bh',
                                              correction_alpha=SIG_ALPHA,
                                              verbose=False)
        compare_train_df.rename(columns={'identifier': 'gene'}, inplace=True)
        compare_train_df['nlog10_p'] = -np.log10(compare_train_df.corr_pval)
        
        ax = axarr[row_ix, col_ix]
        sns.scatterplot(data=compare_train_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                        hue_order=[False, True], ax=ax)
        # add vertical line at 0
        ax.axvline(x=0, linestyle='--', color='black', linewidth=1.25)
        # add horizontal line at statistical significance threshold
        l = ax.axhline(y=-np.log10(SIG_ALPHA), linestyle='--', linewidth=1.25)
        # label horizontal line with significance threshold
        # (matplotlib makes this fairly difficult, sadly)
        ax.text(0.6, -np.log10(SIG_ALPHA)+0.01,
                r'$\mathbf{{\alpha = {}}}$'.format(SIG_ALPHA),
                va='center', ha='center', color=l.get_color(),
                backgroundcolor=ax.get_facecolor())
        # NOTE compare_results function takes df2 - df1, so we have to invert them here
        ax.set_xlabel('AUPR({}) - AUPR({})'.format(train_2, train_1))
        ax.set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
        ax.set_xlim((-0.75, 0.75))
        ax.set_ylim((0, 7))
        ax.legend(title=r'Reject $H_0$', loc='upper left')
        ax.set_title(r'{} PCs, {} vs. {}'.format(n_dims, train_2, train_1),
                     size=13, pad=10)

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand in inkscape if necessary
        text_labels = label_points(compare_train_df['delta_mean'],
                                   compare_train_df['nlog10_p'],
                                   compare_train_df.gene,
                                   compare_train_df.reject_null,
                                   ax)
        adjust_text(text_labels,
                    ax=ax, 
                    expand_text=(1., 1.),
                    lim=5)
        
plt.suptitle('Mutation prediction, comparison of data types', size=16)
plt.tight_layout(w_pad=2, h_pad=2)
plt.subplots_adjust(top=0.94)

