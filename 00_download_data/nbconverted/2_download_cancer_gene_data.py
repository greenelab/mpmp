#!/usr/bin/env python
# coding: utf-8

# ## Download cancer gene sets
# 
# We want to download the set of cancer-associated genes from the [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/cosmic/census), and from [Bailey et al. 2018](https://www.sciencedirect.com/science/article/pii/S009286741830237X), in order to use these genes in our experiments as a comparison/complement to the Vogelstein et al. gene set.

# In[1]:


from pathlib import Path

import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Download COSMIC CGC data
# 
# We downloaded the original CGC data directly from the Sanger Institute website linked above - you need to create an account there to download the .tsv file, so we can't do it programmatically.

# In[2]:


cosmic_df = pd.read_csv(
    cfg.cosmic_raw_file, sep='\t', index_col=0
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


# In[3]:


print(cosmic_df['Role in Cancer'].unique())

# if a gene is annotated as an oncogene/TSG and a fusion gene, just
# get rid of the fusion component
# we'll resolve the dual annotated oncogene/TSG genes later
cosmic_df['Role in Cancer'] = cosmic_df['Role in Cancer'].str.replace(', fusion', '')
print(cosmic_df['Role in Cancer'].unique())


# ### Download Bailey et al. data
# 
# This is a supplementary table from [the TCGA Pan-Cancer Atlas driver gene analysis](https://www.sciencedirect.com/science/article/pii/S009286741830237X). The table contains genes identified as cancer drivers by taking the consensus of existing driver identification methods, in addition to manual curation as described in the paper. The table also contains oncogene/TSG predictions for these genes, using the [20/20+ method](https://2020plus.readthedocs.io/en/latest/).
# 
# This table (Excel file) was also directly downloaded from the paper's supplementary data, as Cell doesn't seem to provide a straightforward API (to my knowledge anyway).

# In[4]:


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

print(class_df.shape)
class_df.head()


# In[5]:


bailey_df = (
    class_df[((class_df.Cancer == 'PANCAN') &
             (~class_df.classification.isna()))]
      .set_index('Gene')
).copy()

# this is the best classification we have to go on for these genes, so if
# a gene is labeled as "possible X", we'll just consider it X
bailey_df['classification'] = (
    bailey_df['classification'].str.replace('possible ', '')
                                   .replace('tsg', 'TSG')
                                   .replace('oncogene', 'Oncogene')
)

print(bailey_df.shape)
bailey_df.head()


# In[6]:


from pandas.api.types import is_datetime64_any_dtype

# make sure no gene names were converted to dates, since excel does this sometimes
assert not is_datetime64_any_dtype(bailey_df.index)


# ### Load Vogelstein et al. data
# 
# This data originally came from [Vogelstein et al. 2013](https://www.science.org/doi/10.1126/science.1235122), and is available as a tsv in [the `pancancer` repo](https://github.com/greenelab/pancancer/blob/master/data/vogelstein_cancergenes.tsv). Oncogene/TSG annotations are also 20/20+ predictions.

# In[7]:


import mpmp.utilities.data_utilities as du

vogelstein_df = du.load_vogelstein()

print(vogelstein_df.shape)
vogelstein_df.head()


# ### Clean up oncogene/TSG annotations and merge datasets
# 
# We want as many genes as possible to be annotated as _either_ an oncogene or TSG, so we know whether to use copy gain or copy loss data to define relevant CNV info. 
# 
# So, here, we will:
# 
# 1. Drop genes that are annotated only as fusion genes (since we're not calling fusions at this time)  
# 2. Try to resolve genes that are annotated as both oncogene/TSG (usually context/cancer type specific) into their most likely pan-cancer category  
# 3. For genes that can't be resolved confidently, we'll keep them as "oncogene, TSG" and run our scripts using both copy gain and copy loss downstream.
# 4. Merge the three datasets together, ensuring that annotations are concordant (or resolving them if not)

# In[8]:


# first merge bailey and vogelstein datasets
# these are both "confidently" annotated (everything is either an oncogene or TSG),
# so we don't need to worry about resolving ambiguous annotations
print(vogelstein_df.classification.unique())
vogelstein_df.head()


# In[9]:


print(bailey_df.classification.unique())
bailey_df.head()


# In[10]:


# first merge dataframes, then resolve classifications
vogelstein_bailey_df = (
    vogelstein_df.loc[:, ['gene', 'classification']]
      .set_index('gene')
      .merge(bailey_df.loc[:, ['classification']],
             how='outer', left_index=True, right_index=True)
      .rename(columns={'classification_x': 'vogelstein_classification',
                       'classification_y': 'bailey_classification'})
)

print(vogelstein_bailey_df.shape)
vogelstein_bailey_df.head()


# In[11]:


def merge_classifications(row):
    if row['vogelstein_classification'] == row['bailey_classification']:
        return row['vogelstein_classification']
    elif pd.isna(row['vogelstein_classification']):
        return row['bailey_classification']
    elif pd.isna(row['bailey_classification']):
        return row['vogelstein_classification']
    elif row['vogelstein_classification'] != row['bailey_classification']:
        # if the datasets disagree, we can resolve these manually
        # or just run them as "oncogene, TSG"
        return 'check'
    else:
        # not sure how this would happen
        return pd.NA
    
vogelstein_bailey_df['classification'] = (
    vogelstein_bailey_df.apply(merge_classifications, axis='columns')
)

print(vogelstein_bailey_df.classification.unique())
vogelstein_bailey_df.head()


# In[12]:


# examples where the datasets disagree, there shouldn't be too many of them
print(vogelstein_bailey_df[vogelstein_bailey_df.classification == 'check'].shape)
vogelstein_bailey_df[vogelstein_bailey_df.classification == 'check']


# In[13]:


# COSMIC CGC classifies DNMT3A as a TSG at the pan-cancer level,
# so we'll go with that
# https://cancer.sanger.ac.uk/cosmic/census-page/DNMT3A
vogelstein_bailey_df.loc['DNMT3A', 'classification'] = 'TSG'

# COSMIC CGC says JAK1 can act as either a TSG or an oncogene, so
# we'll run it as both
# https://cancer.sanger.ac.uk/cosmic/census-page/JAK1
vogelstein_bailey_df.loc['JAK1', 'classification'] = 'Oncogene, TSG'

# SMARCA4 has been characterized as both a tumor suppressor and an
# oncogene, depending on context
# https://www.nature.com/articles/s41388-021-01875-6
# https://www.frontiersin.org/articles/10.3389/fimmu.2021.762598/full
vogelstein_bailey_df.loc['SMARCA4', 'classification'] = 'Oncogene, TSG'

# WT1 can also act as either a TSG or an oncogene depending on the
# cancer type/context:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4352850/
# or even within the same cancer type:
# https://www.nature.com/articles/2404624
vogelstein_bailey_df.loc['WT1', 'classification'] = 'Oncogene, TSG'


vogelstein_bailey_df.drop(
    columns=['vogelstein_classification', 'bailey_classification'],
    inplace=True
)
print(vogelstein_bailey_df.classification.unique())
vogelstein_bailey_df.head()


# In[14]:


# now merge with cosmic CGC genes
merged_df = (vogelstein_bailey_df
  .merge(cosmic_df.loc[:, ['Role in Cancer']],
         how='outer', left_index=True, right_index=True)
  .rename(columns={'classification': 'vb_classification',
                   'Role in Cancer': 'cosmic_classification'})
)
merged_df['cosmic_classification'] = (
    merged_df['cosmic_classification'].str.replace('oncogene', 'Oncogene')
)

print(merged_df.shape)
merged_df.head()


# In[15]:


def merge_all(row):
    if row['vb_classification'] == row['cosmic_classification']:
        return row['vb_classification']
    elif pd.isna(row['vb_classification']):
        return row['cosmic_classification']
    elif pd.isna(row['cosmic_classification']):
        return row['vb_classification']
    elif row['cosmic_classification'] == 'Oncogene, TSG':
        # for ambiguous cosmic examples, just go with the
        # bailey/vogelstein annotation
        return row['vb_classification']
    elif row['vb_classification'] != row['cosmic_classification']:
        # if the datasets disagree, go with the cosmic annotation
        # these are typically manually curated and likely to be more
        # generally applicable
        return row['cosmic_classification']
    else:
        # not sure how this would happen
        return pd.NA
    
merged_df['classification'] = (
    merged_df.apply(merge_all, axis='columns')
)

print(merged_df.classification.unique())
merged_df.head()


# In[16]:


(merged_df
  .loc[:, 'classification']
  .rename_axis('gene')
  .to_csv(cfg.merged_cancer_genes, sep='\t')
)

