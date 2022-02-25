#!/usr/bin/env python
# coding: utf-8

# ## Explore overlap between cancer gene sets
# 
# We want to download the set of cancer-associated genes from the [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/cosmic/census), in order to use these genes in our experiments as a comparison/complement to the Vogelstein et al. gene set.

# In[1]:


from pathlib import Path

import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# load datasets
vogelstein_df = du.load_vogelstein()
vogelstein_df.head()


# In[3]:


cosmic_df = du.load_cosmic()
cosmic_df.head()


# In[4]:


# load Bailey et al. data from excel file
# this is the same as the code in 00_download_data/2_download_cancer_gene_set.ipynb
class_df = pd.read_excel(
    cfg.bailey_raw_file,
    engine='openpyxl', sheet_name='Table S1', index_col='KEY', header=3
)
class_df.drop(
    class_df.columns[class_df.columns.str.contains('Unnamed')],
    axis=1, inplace=True
)
class_df.rename(columns={'Tumor suppressor or oncogene prediction (by 20/20+)':
                         'classification'},
                inplace=True)

bailey_df = (
    class_df[((class_df.Cancer == 'PANCAN') &
             (~class_df.classification.isna()))]
).copy()

bailey_df.head()


# ### Overlap between COSMIC/Bailey/Vogelstein
# 
# Is COSMIC a strict subset of the Bailey and Vogelstein cancer driver datasets? Or are there genes in the latter two that are not in COSMIC?

# In[5]:


vogelstein_genes = set(vogelstein_df.gene.values)
cosmic_genes = set(cosmic_df.gene.values)
bailey_genes = set(bailey_df.Gene.values)


# In[6]:


from venn import venn
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('white')

label_map = {
    'cosmic': cosmic_genes,
    'bailey': bailey_genes,
    'vogelstein': vogelstein_genes
}
venn(label_map)
plt.title('Overlap between cancer gene sets', size=13)

