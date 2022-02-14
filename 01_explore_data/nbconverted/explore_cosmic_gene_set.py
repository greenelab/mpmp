#!/usr/bin/env python
# coding: utf-8

# In[14]:


from pathlib import Path

import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


cosmic_df = pd.read_csv(
    cfg.cosmic_genes_file, sep='\t', index_col=0
)

cosmic_df = cosmic_df[
    # use only tier 1 genes
    ((cosmic_df.Tier == 1) &
    # drop genes without a catalogued somatic mutation
     (cosmic_df.Somatic == 'yes') &
    # drop genes that are only observed in cancer as fusions
    # (we're not calling fusion genes in our mutation data)
     (cosmic_df['Role in Cancer'] != 'fusion'))
].copy()
     
print(cosmic_df.shape)
cosmic_df.head()


# ### Clean up the oncogene/TSG annotations
# 
# We want as many genes as possible to be annotated as _either_ an oncogene or TSG, so we know whether to use copy gain or copy loss data to define relevant CNV info. 
# 
# So, here, we will:
# 
# 1) drop genes that are annotated only as fusion genes (since we're not calling fusions at this time)  
# 2) try to resolve genes that are annotated as both oncogene/TSG (usually context/cancer type specific) into their most likely pan-cancer category  
# 3) for genes that can't be resolved confidently, we'll keep them as "oncogene, TSG" and run our scripts using both copy gain and copy loss downstream.

# In[3]:


print(cosmic_df['Role in Cancer'].unique())

# if a gene is annotated as an oncogene/TSG and a fusion gene, just
# get rid of the fusion component
cosmic_df['Role in Cancer'] = cosmic_df['Role in Cancer'].str.replace(', fusion', '')
print(cosmic_df['Role in Cancer'].unique())


# In[4]:


# how to resolve genes annotated as both oncogene and TSG?
cosmic_dual_df = cosmic_df[cosmic_df['Role in Cancer'] == 'oncogene, TSG']
print(cosmic_dual_df.shape)
print(cosmic_dual_df.index)
cosmic_dual_df.head()


# In[5]:


# load Bailey et al. data
# supplementary table from https://www.sciencedirect.com/science/article/pii/S009286741830237X
# this contains oncogene/TSG predictions for genes/cancer types using 20/20+ classifier
class_df = pd.read_excel(
    cfg.data_dir / '1-s2.0-S009286741830237X-mmc1.xlsx', 
    engine='openpyxl', sheet_name='Table S1', index_col='KEY', header=3
)
class_df.rename(columns={'Tumor suppressor or oncogene prediction (by 20/20+)':
                         'classification'},
                inplace=True)

print(class_df.shape)
class_df.head()


# In[6]:


# TODO: should we use genes that aren't labeled as pan-cancer drivers here?
bailey_predicted_df = (
    class_df[((class_df.Cancer == 'PANCAN') &
             (class_df.Gene.isin(cosmic_dual_df.index)) &
             (~class_df.classification.isna()))]
).copy()

# this is the best classification we have to go on in these cases, so if something
# is labeled as "possible X", we'll just consider it X
bailey_predicted_df['classification'] = (
    bailey_predicted_df['classification'].str.replace('possible ', '')
                                         .replace('tsg', 'TSG')
)

print(bailey_predicted_df.shape)
bailey_predicted_df.head()


# In[7]:


cosmic_clean_df = (cosmic_dual_df.loc[:, ['Role in Cancer']]
    .merge(bailey_predicted_df.loc[:, ['Gene', 'classification']],
           left_index=True, right_on='Gene')
    .set_index('Gene')
)
cosmic_clean_df.head(10)


# In[8]:


# TP53 should be labeled as "oncogene, TSG"
# (see, e.g. https://www.nature.com/articles/cdd201553)
cosmic_df.loc[cosmic_df.index == 'TP53', ['Role in Cancer']]


# In[9]:


cosmic_df.loc[cosmic_clean_df.index, 'Role in Cancer'] = cosmic_clean_df.classification
print(cosmic_df.shape)
cosmic_df.head()


# In[10]:


# TP53 should be labeled as TSG now, even though it can act as an oncogene sometimes
cosmic_df.loc[cosmic_df.index == 'TP53', ['Role in Cancer']]


# In[11]:


# save cleaned up annotations to use in our classifiers
cosmic_df.loc[:, ['Role in Cancer']].to_csv(
    cfg.cosmic_with_annotations, sep='\t'
)


# ### Overlap between COSMIC/Bailey/Vogelstein
# 
# Is COSMIC a strict subset of the Bailey and Vogelstein cancer driver datasets? Or are there genes in the latter two that are not in COSMIC?

# In[19]:


cosmic_genes = set(cosmic_df.index.values)
bailey_genes = set(class_df[class_df.Cancer == 'PANCAN'].Gene.values)

vogelstein_df = du.load_vogelstein()
vogelstein_genes = set(vogelstein_df.gene.values)


# In[26]:


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

