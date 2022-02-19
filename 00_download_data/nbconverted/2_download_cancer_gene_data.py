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
# This table (Excel file) was also directly downloaded from the paper's supplementary data, as Cell doesn't seem to provide a straightforward API (that I'm able to find).

# In[4]:


class_df = pd.read_excel(
    cfg.data_dir / '1-s2.0-S009286741830237X-mmc1.xlsx', 
    engine='openpyxl', sheet_name='Table S1', index_col='KEY', header=3
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


# ### Load Vogelstein et al. data
# 
# This data originally came from [Vogelstein et al. 2013](https://www.science.org/doi/10.1126/science.1235122). Oncogene/TSG annotations also come from 20/20+ predictions.

# In[6]:


import mpmp.utilities.data_utilities as du

vogelstein_df = du.load_vogelstein()

print(vogelstein_df.shape)
vogelstein_df.head()

