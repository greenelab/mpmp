#!/usr/bin/env python
# coding: utf-8

# ## Download and explore COSMIC cancer genes
# 
# We want to download the set of cancer-associated genes from the [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/cosmic/census), in order to use these genes in our experiments as a comparison/complement to the Vogelstein et al. gene set.

# In[1]:


from pathlib import Path

import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load and process raw COSMIC data
# 
# This has to be downloaded directly from the CGC website linked above - you need an account there to download the .tsv file, so we can't do it programmatically.

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


# get genes labeled by Bailey et al. as pan-cancer drivers, that are
# also in COSMIC set and have annotations
bailey_predicted_df = (
    class_df[((class_df.Cancer == 'PANCAN') &
             (class_df.Gene.isin(cosmic_dual_df.index)) &
             (~class_df.classification.isna()))]
).copy()

# this is the best annotation we have to go on in these cases, so if something
# is labeled as "possible X", we'll just consider it X
bailey_predicted_df['classification'] = (
    bailey_predicted_df['classification'].str.replace('possible ', '')
                                         .replace('tsg', 'TSG')
                                         .replace('oncogene', 'Oncogene')
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


# TP53 should be labeled as TSG now, since this is generally how it behaves
# (even though it can act as an oncogene under specific conditions)
cosmic_df.loc[cosmic_df.index == 'TP53', ['Role in Cancer']]


# In[11]:


# save cleaned up annotations to use in our classifiers
cosmic_df['Role in Cancer'] = cosmic_df['Role in Cancer'].str.replace(
    'oncogene', 'Oncogene'
)

(cosmic_df
    .loc[:, ['Role in Cancer']]
    .rename(columns={'Role in Cancer': 'classification'})
    .rename_axis('gene')
    .to_csv(cfg.cosmic_with_annotations, sep='\t')
)


# ### Overlap between COSMIC/Bailey/Vogelstein
# 
# Is COSMIC a strict subset of the Bailey and Vogelstein cancer driver datasets? Or are there genes in the latter two that are not in COSMIC?

# In[12]:


cosmic_genes = set(cosmic_df.index.values)
bailey_genes = set(class_df[class_df.Cancer == 'PANCAN'].Gene.values)

vogelstein_df = du.load_vogelstein()
vogelstein_genes = set(vogelstein_df.gene.values)


# In[13]:


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


# ### How many COSMIC genes have at least one valid cancer type?
# 
# Or, in other words, how many genes will we need to run our classifiers for?

# In[17]:


import sys

from mpmp.data_models.tcga_data_model import TCGADataModel

def calculate_gene_count(genes_df, verbose):
    """For a set of data types, calculate the number of valid genes."""
    gene_list = []
    sample_info_df = du.load_sample_info('expression')
    tcga_data = TCGADataModel(seed=cfg.default_seed, verbose=verbose)
    for gene_ix, gene_series in genes_df.iterrows():
        try:
            if gene_ix % 10 == 0:
                print(gene_ix, '/', genes_df.shape[0], file=sys.stderr)
            tcga_data.process_data_for_gene(gene_series.gene,
                                            gene_series.classification,
                                            None)
        except KeyError:
            print('key error:', gene_series.gene)
            continue
        gene_list.append((gene_series.gene, tcga_data.X_df.shape[0]))
    return gene_list


# In[18]:


genes_df = pd.read_csv(cfg.cosmic_with_annotations, sep='\t')
genes_df.head()


# In[20]:


gene_list = calculate_gene_count(genes_df, verbose=False)
nonzero_genes = [g for g, c in gene_list if c > 0]
print('Nonzero genes:', len(nonzero_genes), '/', len(gene_list))

