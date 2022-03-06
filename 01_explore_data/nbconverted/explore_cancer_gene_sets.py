#!/usr/bin/env python
# coding: utf-8

# ## Explore cancer gene sets
# 
# In `00_download_data/2_download_cancer_gene_data.ipynb`, we downloaded and integrated several different sets of cancer-associated genes. That script contains info about which sources we're using to get our list of cancer genes, how we're labeling them as oncogene/TSG, and how the data sources are similar/different.
# 
# In this script, we want to explore the similarities and differences between these gene sets in more detail. We have seen that the results of our classification experiments are somewhat dependent on which gene set or sets we decide to use, so understanding the similarities and differences between the gene sets will help us to put these results in context.

# In[1]:


import sys
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


# ### Enrichment analysis of gene sets
# 
# Here, we want to do a GO molecular function enrichment analysis of the gene sets we're using. In particular, we want to compare enriched functions for the Vogelstein et al. and "merged" (Vogelstein + COSMIC + Bailey) cancer gene sets, since the classification results we see for these gene sets are so different.
# 
# The code below is based on the `goatools` example vignette, which is a Python package we use to analyze functional enrichment: https://github.com/tanghaibao/goatools/blob/main/notebooks/goea_nbt3102.ipynb

# In[7]:


# download ontology data
cfg.go_data_dir.mkdir(exist_ok=True)
obo_file = cfg.go_data_dir / 'go-basic.obo'

if not obo_file.exists():
    from goatools.base import download_go_basic_obo
    obo_fname = download_go_basic_obo()
    Path(obo_fname).replace(obo_file)
else:
    print('Ontology data file already exists')


# In[8]:


# download gene-GO associations
gene2go_file = cfg.go_data_dir / 'gene2go.gz'

if not gene2go_file.exists():
    from goatools.base import download_ncbi_associations
    gene2go_fname = download_ncbi_associations()
    Path(gene2go_fname).replace(gene2go_file)
else:
    print('Associations data file already exists')


# In[9]:


# load ontology structure
from goatools.obo_parser import GODag

obodag = GODag(str(obo_file))


# In[10]:


from goatools.anno.genetogo_reader import Gene2GoReader

# Read NCBI's gene2go. Store annotations in a list of namedtuples
objanno = Gene2GoReader(str(gene2go_file), taxids=[9606])

# Get namespace2association where:
#    namespace is:
#        BP: biological_process               
#        MF: molecular_function
#        CC: cellular_component
#    assocation is a dict:
#        key: NCBI GeneID
#        value: A set of GO IDs associated with that gene
ns2assoc = objanno.get_ns2assc()

for nspc, id2gos in ns2assoc.items():
    print("{NS} {N:,} annotated human genes".format(NS=nspc, N=len(id2gos)))


# In[11]:


# convert NCBI Gene query results for human genes to Python module
# see: https://github.com/tanghaibao/goatools/blob/main/notebooks/background_genes_ncbi.ipynb
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py

ncbi_tsv = str(cfg.go_data_dir / 'gene_result.txt')
output_py = 'genes_ncbi_9606_proteincoding.py'
ncbi_tsv_to_py(ncbi_tsv, output_py)


# In[12]:


# load human background gene set
from genes_ncbi_9606_proteincoding import GENEID2NT as GeneID2nt

print(len(GeneID2nt))
print(list(GeneID2nt.items())[0])


# In[13]:


# initialize GO enrichment analysis object
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

goea = GOEnrichmentStudyNS(
    GeneID2nt.keys(), # list of human protein-coding genes
    ns2assoc, # gene/GO term associations
    obodag, # ontology structure
    propagate_counts=False,
    alpha=0.05,
    methods=['fdr_bh']
)


# In[14]:


# we need to convert gene symbols to entrez IDs
from mpmp.utilities.tcga_utilities import get_symbol_map

symbol_to_entrez, old_to_new_entrez = get_symbol_map()

def gene_names_to_ids(gene_names):
    gene_ids = []
    for gene in gene_names:
        try:
            gene_id = symbol_to_entrez[gene]
            if gene_id in old_to_new_entrez.keys():
                gene_id = old_to_new_entrez[gene_id]
            gene_ids.append(gene_id)
        except KeyError:
            print('Gene {} not in ID map'.format(gene), file=sys.stderr)
            continue
    return gene_ids


# In[15]:


# get entrez IDs for Vogelstein genes        
vogelstein_ids = gene_names_to_ids(vogelstein_df.gene.unique())
print(len(vogelstein_ids))
vogelstein_ids[:5]


# In[16]:


# run GO enrichment analysis for Vogelstein genes
vogelstein_results = goea.run_study(vogelstein_ids)
vogelstein_sig_results = [r for r in vogelstein_results if r.p_fdr_bh < 0.05]


# In[17]:


# save to output directory
cfg.go_output_dir.mkdir(exist_ok=True)

vogelstein_output_file = cfg.go_output_dir / 'vogelstein_enrichment.tsv'
goea.wr_tsv(str(vogelstein_output_file), vogelstein_sig_results)


# In[18]:


# get non-Vogelstein gene set
all_genes = set.union(*list(label_map.values()))
non_vogelstein_genes = all_genes - label_map['vogelstein']
print(len(all_genes))
print(len(non_vogelstein_genes))


# In[19]:


# get entrez IDs for non-Vogelstein genes        
non_vogelstein_ids = gene_names_to_ids(non_vogelstein_genes)
print(len(non_vogelstein_ids))
non_vogelstein_ids[:5]


# In[20]:


# run GO enrichment analysis for non-Vogelstein genes
non_vogelstein_results = goea.run_study(non_vogelstein_ids)
non_vogelstein_sig_results = [r for r in non_vogelstein_results if r.p_fdr_bh < 0.05]


# In[21]:


non_vogelstein_output_file = cfg.go_output_dir / 'non_vogelstein_enrichment.tsv'
goea.wr_tsv(str(non_vogelstein_output_file), non_vogelstein_sig_results)


# In[22]:


# get entrez IDs for all gene set
all_ids = gene_names_to_ids(all_genes)
print(len(all_ids))
all_ids[:5]


# In[23]:


# run GO enrichment analysis for non-Vogelstein genes
all_results = goea.run_study(all_ids)
all_sig_results = [r for r in all_results if r.p_fdr_bh < 0.05]


# In[24]:


all_output_file = cfg.go_output_dir / 'merged_enrichment.tsv'
goea.wr_tsv(str(all_output_file), all_sig_results)


# ### Overlap of enrichment results
# 
# We want to see if there are GO terms that are enriched in one gene set (Vogelstein/non-Vogelstein) but not the other. The hope is this will tell us something about why the gene sets result in such different (relative) performance between data types.

# In[25]:


go_category = 'MF'

vogelstein_enrichment_df = (
    pd.read_csv(vogelstein_output_file, sep='\t')
      .query("NS == @go_category")
      .reset_index(drop=True)
)

print(vogelstein_enrichment_df.shape)
vogelstein_enrichment_df.head()


# In[26]:


non_vogelstein_enrichment_df = (
    pd.read_csv(non_vogelstein_output_file, sep='\t')
      .query("NS == @go_category")
      .reset_index(drop=True)
)

print(non_vogelstein_enrichment_df.shape)
non_vogelstein_enrichment_df.head()


# In[27]:


sns.set_style('white')

vogelstein_set = set(vogelstein_enrichment_df['# GO'].values)
non_vogelstein_set = set(non_vogelstein_enrichment_df['# GO'].values)

go_term_map = {
    'vogelstein': vogelstein_set,
    'non_vogelstein': non_vogelstein_set,
}
venn(go_term_map)
plt.title('Overlap between enriched GO {} terms'.format(go_category), size=13)


# In[28]:


vogelstein_only_terms = (
    vogelstein_set - non_vogelstein_set
)
print(vogelstein_enrichment_df[vogelstein_enrichment_df['# GO'].isin(vogelstein_only_terms)].shape)
vogelstein_enrichment_df[vogelstein_enrichment_df['# GO'].isin(vogelstein_only_terms)].head(10)


# In[29]:


non_vogelstein_only_terms = (
    non_vogelstein_set - vogelstein_set
)
print(non_vogelstein_enrichment_df[non_vogelstein_enrichment_df['# GO'].isin(non_vogelstein_only_terms)].shape)
non_vogelstein_enrichment_df[non_vogelstein_enrichment_df['# GO'].isin(non_vogelstein_only_terms)].head(10)


# In[30]:


overlap_terms = (
    non_vogelstein_set & vogelstein_set
)
print(non_vogelstein_enrichment_df[non_vogelstein_enrichment_df['# GO'].isin(overlap_terms)].shape)
non_vogelstein_enrichment_df[non_vogelstein_enrichment_df['# GO'].isin(overlap_terms)].head(10)


# Results are summarized in the Google slides here: https://docs.google.com/presentation/d/1AVK7XVRHq3xkeH7U1413jQLh79zDgS6Q4VEPsXcDTQU/edit?usp=sharing
# 
# Comparing the Vogelstein-only and non-Vogelstein-only terms, for the MF enrichment I don't honestly see a huge difference. Both gene sets are enriched for lots of "DNA binding" and "protein binding"-type terms. The non-Vogelstein gene set has some terms related to transcriptional activation/repression which the Vogelstein gene set doesn't have, so maybe these genes could induce changes to transcription which are picked up more easily in the gene expression dataset.
# 
# Looking at BP enrichment, we can see that the Vogelstein-only genes are enriched for MAPK signaling genes, and the non-Vogelstein genes are enriched for cell cycle genes and transcriptional regulators.

# In[31]:


from goatools.godag_plot import plot_gos

# select some terms from the non-vogelstein genes to visualize
# if we take all of them we get an unreadable mess
terms_to_plot = [
    'GO:0000981', # DNA-binding TF activity
    'GO:0003712', # transcription coregulator activity
    'GO:0001227', # DNA-binding transcription repressor activity
    'GO:0019904', # protein domain specific binding activity
    'GO:1990837', # double-stranded DNA binding
]

plot_gos("non_vogelstein_tf.png", terms_to_plot, obodag, goea_results=non_vogelstein_results)


# From the `goatools` vignette:
# 
# ```
# GO terms colored by P-value:
# pval < 0.005 (light red)
# pval < 0.01 (light orange)
# pval < 0.05 (yellow)
# pval > 0.05 (grey) Study terms that are not statistically significant
# ```

# ![image](non_vogelstein_tf.png)
