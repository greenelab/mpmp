#!/usr/bin/env python
# coding: utf-8

# ## Plot cancer type prediction results

# Notebook parameters:
# * FILTER_CANCER_TYPES (list): cancer types to filter box plots to
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
results_dir = Path(cfg.results_dir, 'cancer_type').resolve()
# set significance cutoff after FDR correction
SIG_ALPHA = 0.001


# In[3]:


# load raw data
results_df = au.load_stratified_prediction_results(results_dir, 'cancer_type')
print(results_df.shape)
results_df.head()


# In[4]:


# set this variable to filter plot to certain cancer types
# if not included, plot all 33 of them
# FILTER_CANCER_TYPES = ['LUAD', 'LUSC', 'THCA']
FILTER_CANCER_TYPES = None
filtered_df = results_df[
    (results_df.signal == 'signal') &
    (results_df.data_type == 'test')
].copy()
if FILTER_CANCER_TYPES is not None:
    filtered_df = filtered_df[
        (filtered_df.cancer_type.isin(FILTER_CANCER_TYPES))
    ]
else:
    filtered_df.sort_values(by='cancer_type', inplace=True)

sns.set({'figure.figsize': (20, 5)})
sns.boxplot(data=filtered_df, x='cancer_type', y='aupr', hue='training_data',
            hue_order=np.sort(filtered_df.training_data.unique()))
plt.xlabel('TCGA cancer type')
plt.ylabel('AUPR')
plt.legend(title='Training data type')
plt.title('Performance distribution for cancer type prediction, expression vs. methylation data')


# In[5]:


expression_df = (
    results_df[results_df.training_data == 'expression']
        .drop(columns=['training_data'])
)
expression_results_df = au.compare_results(expression_df,
                                           identifier='cancer_type',
                                           metric='aupr',
                                           correction=True,
                                           correction_method='fdr_bh',
                                           correction_alpha=SIG_ALPHA,
                                           verbose=True)
expression_results_df.sort_values(by='p_value').head(n=10)


# In[6]:


methylation_df = (
    results_df[results_df.training_data == 'methylation']
        .drop(columns=['training_data'])
)
methylation_results_df = au.compare_results(methylation_df,
                                            identifier='cancer_type',
                                            metric='aupr',
                                            correction=True,
                                            correction_method='fdr_bh',
                                            correction_alpha=SIG_ALPHA,
                                            verbose=True)
methylation_results_df.sort_values(by='p_value').head(n=10)


# In[7]:


expression_results_df['nlog10_p'] = -np.log10(expression_results_df.corr_pval)
methylation_results_df['nlog10_p'] = -np.log10(methylation_results_df.corr_pval)

sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)

# plot cancer type prediction from expression, in a volcano-like plot
sns.scatterplot(data=expression_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                hue_order=[False, True], ax=axarr[0])
# add vertical line at 0
axarr[0].axvline(x=0, linestyle=':', color='black')
# add horizontal line at statistical significance threshold
l = axarr[0].axhline(y=-np.log10(SIG_ALPHA), linestyle=':')
# label horizontal line with significance threshold
# (matplotlib makes this fairly difficult, sadly)
axarr[0].text(0.9, -np.log10(SIG_ALPHA)+0.3,
              r'$\alpha = {}$'.format(SIG_ALPHA),
              va='center', ha='center', color=l.get_color(),
              backgroundcolor=axarr[0].get_facecolor())
axarr[0].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[0].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[0].set_xlim((-0.2, 1.0))
axarr[0].set_ylim((0, expression_results_df.nlog10_p.max() + 5))
axarr[0].legend(title=r'Reject $H_0$', loc='upper left')
axarr[0].set_title(r'Cancer type prediction, expression data')

# plot cancer type prediction from methylation, same as above
sns.scatterplot(data=methylation_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                hue_order=[False, True], ax=axarr[1])
axarr[1].axvline(x=0, linestyle=':', color='black')
l = axarr[1].axhline(y=-np.log10(SIG_ALPHA), linestyle=':')
axarr[1].text(0.9, -np.log10(SIG_ALPHA)+0.3,
              r'$\alpha = {}$'.format(SIG_ALPHA),
              va='center', ha='center', color=l.get_color(),
              backgroundcolor=axarr[0].get_facecolor())
axarr[1].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[1].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[1].set_xlim((-0.2, 1.0))
axarr[1].set_ylim((0, methylation_results_df.nlog10_p.max() + 5))
axarr[1].legend(title=r'Reject $H_0$', loc='upper left')
axarr[1].set_title(r'Cancer type prediction, methylation data')

def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        text_labels.append(
            ax.text(point['x']+.005, point['y']+.2, str(point['gene']))
        )
    return text_labels

# label cancer types and adjust text to not overlap
# automatic alignment isn't perfect, can align by hand in inkscape if necessary
text_labels_expression = label_points(expression_results_df['delta_mean'],
                                      expression_results_df['nlog10_p'],
                                      expression_results_df.identifier,
                                      expression_results_df.reject_null,
                                      axarr[0])
adjust_text(text_labels_expression, ax=axarr[0])

text_labels_methylation = label_points(methylation_results_df['delta_mean'],
                                       methylation_results_df['nlog10_p'],
                                       methylation_results_df.identifier,
                                       methylation_results_df.reject_null,
                                       axarr[1])
adjust_text(text_labels_methylation, ax=axarr[1])


# In[8]:


compare_results_df = au.compare_results(methylation_df,
                                        pancancer_df=expression_df,
                                        identifier='cancer_type',
                                        metric='aupr',
                                        correction=True,
                                        correction_method='fdr_bh',
                                        correction_alpha=SIG_ALPHA,
                                        verbose=True)
compare_results_df.head()


# In[9]:


compare_results_df['nlog10_p'] = -np.log10(compare_results_df.corr_pval)

sns.set({'figure.figsize': (12, 8)})
sns.scatterplot(data=compare_results_df, x='delta_mean', y='nlog10_p', hue='reject_null')
plt.axvline(x=0, linestyle=':', color='black')
l = plt.axhline(y=-np.log10(SIG_ALPHA), linestyle=':')
plt.text(0.9, -np.log10(SIG_ALPHA)+0.05,
         r'$\alpha = {}$'.format(SIG_ALPHA),
         va='center', ha='center', color=l.get_color(),
         backgroundcolor=plt.gca().get_facecolor())
plt.xlabel('AUPR(expression) - AUPR(methylation)')
plt.ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
plt.xlim((-1.0, 1.0))
plt.legend(title=r'Reject $H_0$', loc='upper left')
plt.title(r'Cancer type prediction, expression vs. methylation')

def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        if point['y'] > -np.log10(0.1):
            text_labels.append(
                ax.text(point['x']+.005, point['y']+.1, str(point['gene']))
            )
        elif point['x'] < 0.0:
            # align these left, otherwise can't read
            text_labels.append(
                ax.text(point['x']-.01, point['y']+.1, str(point['gene']),
                        ha='right', va='bottom')
            )
    return text_labels

text_labels = label_points(compare_results_df['delta_mean'],
                           compare_results_df['nlog10_p'],
                           compare_results_df.identifier,
                           compare_results_df.reject_null,
                           plt.gca())
adjust_text(text_labels, ax=plt.gca())


# ## Confusion matrix

# In[10]:


import os

import mpmp.utilities.data_utilities as du

preds_dir = os.path.join(cfg.repo_root, 'results_preds', 'cancer_type')
sample_info_df = du.load_sample_info()

preds_expression_df = au.load_preds_to_matrix(preds_dir, sample_info_df,
                                              training_data='expression')
print(preds_expression_df.shape)
preds_expression_df.iloc[:5, :5]


# In[23]:


sns.set({'figure.figsize': (15, 10)})
ax = sns.heatmap(preds_expression_df, cbar_kws={'label': 'Predicted probability of positive label, averaged over samples'})
# this is needed to increase colorbar label font size
# https://stackoverflow.com/a/48587137
ax.figure.axes[-1].yaxis.label.set_size(14)
plt.xlabel('True cancer type label', size=14)
plt.ylabel('Positive label used to train classifier', size=14)
plt.title('Cancer type confusion matrix, gene expression data', size=14, pad=14)
plt.tight_layout()


# In[13]:


preds_methylation_df = au.load_preds_to_matrix(preds_dir, sample_info_df,
                                               training_data='methylation')
print(preds_methylation_df.shape)
preds_methylation_df.iloc[:5, :5]


# In[24]:


sns.set({'figure.figsize': (15, 10)})
ax = sns.heatmap(preds_methylation_df, cbar_kws={'label': 'Predicted probability of positive label, averaged over samples'})
# this is needed to increase colorbar label font size
# https://stackoverflow.com/a/48587137
ax.figure.axes[-1].yaxis.label.set_size(14)
plt.xlabel('True cancer type label', size=14)
plt.ylabel('Positive label used to train classifier', size=14)
plt.title('Cancer type confusion matrix, methylation data', size=14, pad=14)
plt.tight_layout()


# In[ ]:




