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
import mpmp.utilities.plot_utilities as plu

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# set results directory
drop_results_dir = Path(cfg.results_dirs['mutation'],
                        'dosage_effects',
                        'drop_target',
                        'gene')

control_results_dir = Path(cfg.results_dirs['mutation'],
                           'shuffle_cancer_type',
                           'expression_vogelstein',
                           'gene')


# In[3]:


drop_results_df = au.load_stratified_prediction_results(drop_results_dir, 'gene')
drop_results_df['experiment'] = 'drop_target'
drop_results_df = (drop_results_df
    .drop(columns='training_data')
    .rename(columns={'experiment': 'training_data'})
)

print(drop_results_df.shape)
drop_results_df.head()


# In[4]:


control_results_df = au.load_stratified_prediction_results(control_results_dir, 'gene')
control_results_df['experiment'] = 'control'
control_results_df = (control_results_df
    .drop(columns='training_data')
    .rename(columns={'experiment': 'training_data'})
)

print(control_results_df.shape)
control_results_df.head()


# In[12]:


compare_results_df = au.compare_results(control_results_df,
                                        condition_2_df=drop_results_df,
                                        identifier='identifier',
                                        metric='aupr',
                                        correction=True,
                                        correction_method='fdr_bh',
                                        correction_alpha=0.05,
                                        verbose=True)
compare_results_df['nlog10_p'] = -np.log10(compare_results_df.corr_pval)
print(compare_results_df.shape)
compare_results_df.sort_values(by='corr_pval').head()


# In[28]:


sns.set({'figure.figsize': (8, 6)})
sns.set_style('whitegrid')

ax = plt.gca()

plt.xlim(-0.8, 0.8)
plt.ylim(0, 8)

# add vertical line at 0
ax.axvline(x=0, linestyle='--', linewidth=1.25, color='black')

for alpha in [0.05, 0.01, 0.001]:
    # add horizontal line at statistical significance threshold
    l = ax.axhline(y=-np.log10(alpha), linestyle='--', linewidth=1.25)

    # label horizontal line with significance threshold
    # (matplotlib makes this fairly difficult, sadly)
    ax.text(0.6, -np.log10(alpha)+0.01,
            r'$\mathbf{{\alpha = {}}}$'.format(alpha),
            va='center', ha='center', color=l.get_color(),
            backgroundcolor=ax.get_facecolor())

sns.scatterplot(data=compare_results_df,
                x='delta_mean',
                y='nlog10_p',
                hue='reject_null')

plt.xlabel('AUPR(control) - AUPR(drop target)', size=13)
plt.ylabel(r'$-\log_{10}($adjusted $p$-value$)$', size=13)
plt.legend(title=r'Reject $H_0$', loc='upper left',
           fontsize=13, title_fontsize=13)

