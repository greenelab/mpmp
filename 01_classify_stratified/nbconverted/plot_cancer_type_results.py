#!/usr/bin/env python
# coding: utf-8

# ## Plot cancer type prediction results

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


# In[3]:


# load raw data
results_df = au.load_cancer_type_prediction_results(results_dir, 'cancer_type')
print(results_df.shape)
results_df.head()


# In[4]:


# set this variable to filter plot to certain cancer types
# if not included, plot all 33 of them
# filter_cancer_types = ['LUAD', 'LUSC', 'THCA']
filter_cancer_types = None
if filter_cancer_types is not None:
    filtered_df = results_df[
        (results_df.signal == 'signal') &
        (results_df.data_type == 'test') &
        (results_df.cancer_type.isin(filter_cancer_types))
    ]
else:
    filtered_df = results_df[
        (results_df.signal == 'signal') &
        (results_df.data_type == 'test')
    ].sort_values(by='cancer_type')

sns.set({'figure.figsize': (20, 5)})
sns.boxplot(data=filtered_df, x='cancer_type', y='aupr', hue='training_data',
            hue_order=np.sort(filtered_df.training_data.unique()))
plt.xlabel('TCGA cancer type')
plt.ylabel('AUPR')
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
                                           correction_alpha=0.001,
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
                                            correction_alpha=0.001,
                                            verbose=True)
methylation_results_df.sort_values(by='p_value').head(n=10)


# In[7]:


expression_results_df['nlog10_p'] = -np.log(expression_results_df.corr_pval)
methylation_results_df['nlog10_p'] = -np.log(methylation_results_df.corr_pval)

sns.set({'figure.figsize': (20, 8)})
fig, axarr = plt.subplots(1, 2)
sns.scatterplot(data=expression_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                hue_order=[False, True], ax=axarr[0])
axarr[0].axvline(x=0, linestyle=':', color='black')
axarr[0].axhline(y=-np.log(0.001), linestyle=':')
axarr[0].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[0].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[0].set_xlim((-0.2, 1.0))
axarr[0].set_ylim((0, expression_results_df.nlog10_p.max() + 5))
axarr[0].legend(title=r'Reject $H_0$')
axarr[0].set_title(r'Cancer type prediction, expression data')
sns.scatterplot(data=methylation_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                hue_order=[False, True], ax=axarr[1])
axarr[1].axvline(x=0, linestyle=':', color='black')
axarr[1].axhline(y=-np.log(0.001), linestyle=':')
axarr[1].set_xlabel('AUPR(signal) - AUPR(shuffled)')
axarr[1].set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
axarr[1].set_xlim((-0.2, 1.0))
axarr[1].set_ylim((0, methylation_results_df.nlog10_p.max() + 5))
axarr[1].legend(title=r'Reject $H_0$')
axarr[1].set_title(r'Cancer type prediction, methylation data')

def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        # if point['sig']:
        text_labels.append(
            ax.text(point['x']+.005, point['y']+.2, str(point['gene']))
        )
    return text_labels
    

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
                                        correction_alpha=0.001,
                                        verbose=True)
compare_results_df.head()


# In[9]:


compare_results_df['nlog10_p'] = -np.log(compare_results_df.corr_pval)

sns.set({'figure.figsize': (12, 8)})
sns.scatterplot(data=compare_results_df, x='delta_mean', y='nlog10_p', hue='reject_null')
plt.axvline(x=0, linestyle=':', color='black')
plt.axhline(y=-np.log(0.001), linestyle=':')
plt.xlabel('AUPR(expression) - AUPR(methylation)')
plt.ylabel(r'$-\log_{10}($adjusted $p$-value$)$')
plt.xlim((-1.0, 1.0))
plt.legend(title=r'Reject $H_0$')
plt.title(r'Cancer type prediction, expression vs. methylation')

def label_points(x, y, gene, sig, ax):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'gene': gene, 'sig': sig})
    for i, point in a.iterrows():
        if point['y'] > -np.log(0.1):
            text_labels.append(
                ax.text(point['x']+.005, point['y']+.2, str(point['gene']))
            )
    return text_labels

text_labels = label_points(compare_results_df['delta_mean'],
                           compare_results_df['nlog10_p'],
                           compare_results_df.identifier,
                           compare_results_df.reject_null,
                           plt.gca())
adjust_text(text_labels, ax=plt.gca())

