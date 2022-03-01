#!/usr/bin/env python
# coding: utf-8

# ## Explore cancer gene sets
# 
# We want to download the set of cancer-associated genes from the [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/cosmic/census), in order to use these genes in our experiments as a comparison/complement to the Vogelstein et al. gene set.
# 
# TODO: document in more detail

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


# ### Enrichment analysis of gene sets
# 
# Here, we want to do a GO molecular function enrichment analysis of the gene sets we're using. In particular, we want to compare enriched functions for the Vogelstein et al. and merged cancer gene sets, since the classification results we see for these gene sets are so different.
# 
# The code below mostly follows the `goatools` tutorial here: https://github.com/tanghaibao/goatools/blob/main/notebooks/goea_nbt3102.ipynb

# In[10]:


# download ontology data
cfg.go_data_dir.mkdir(exist_ok=True)
obo_file = cfg.go_data_dir / 'go-basic.obo'

if not obo_file.exists():
    from goatools.base import download_go_basic_obo
    obo_fname = download_go_basic_obo()
    Path(obo_fname).replace(obo_file)
else:
    print('Ontology data file already exists')


# In[13]:


# download gene-GO associations
gene2go_file = cfg.go_data_dir / 'gene2go.gz'

if not gene2go_file.exists():
    from goatools.base import download_ncbi_associations
    gene2go_fname = download_ncbi_associations()
    Path(gene2go_fname).replace(gene2go_file)
else:
    print('Associations data file already exists')


# In[14]:


# load ontology structure
from goatools.obo_parser import GODag

obodag = GODag(str(obo_file))


# In[16]:


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


# In[17]:


# convert NCBI Gene query results for human genes to Python module
# see: https://github.com/tanghaibao/goatools/blob/main/notebooks/background_genes_ncbi.ipynb
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py

ncbi_tsv = str(cfg.go_data_dir / 'gene_result.txt')
output_py = 'genes_ncbi_9606_proteincoding.py'
ncbi_tsv_to_py(ncbi_tsv, output_py)


# In[22]:


# load human background gene set
from genes_ncbi_9606_proteincoding import GENEID2NT as GeneID2nt

print(len(GeneID2nt))
print(list(GeneID2nt.items())[0])


# In[23]:


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


# In[25]:


# we need to convert gene symbols to entrez IDs
from mpmp.utilities.tcga_utilities import get_symbol_map

symbol_to_entrez, old_to_new_entrez = get_symbol_map()
vogelstein_ids = []

for gene in vogelstein_df.gene.unique():
    try:
        gene_id = symbol_to_entrez[gene]
        if gene_id in old_to_new_entrez.keys():
            gene_id = old_to_new_entrez[gene_id]
        vogelstein_ids.append(gene_id)
    except KeyError:
        print('Gene {} not in ID map'.format(gene), file=sys.stderr)
        continue
        
print(len(vogelstein_ids))
vogelstein_ids[:5]


# In[27]:


# run GO enrichment analysis for Vogelstein genes
vogelstein_results = goea.run_study(vogelstein_ids)
vogelstein_sig_results = [r for r in vogelstein_results if r.p_fdr_bh < 0.05]


# In[33]:


goea.wr_tsv('./vogelstein_enrichment.tsv', vogelstein_sig_results)


# In[ ]:




