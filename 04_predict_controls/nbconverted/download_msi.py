#!/usr/bin/env python
# coding: utf-8

# ## Download microsatellite instability groups
# 
# We download pre-computed MSI status information from Firebrowse, as described in the supplement [of this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008878).
# 
# This information exists for 4 cancer types: COAD, STAD, READ, UCEC.

# In[1]:


import os
import pandas as pd
import urllib.request
import tarfile

import mpmp.config as cfg


# In[2]:


# URL locations of zip files containing clinical info
clinical_zip_files = {
    'COADREAD': 'http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/COADREAD/20160128/gdac.broadinstitute.org_COADREAD.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz',
    'STAD': 'http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/STAD/20160128/gdac.broadinstitute.org_STAD.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz',
    'UCEC': 'http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/UCEC/20160128/gdac.broadinstitute.org_UCEC.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz'
}

# where to save extracted clinical tsv files
os.makedirs(cfg.msi_data_dir, exist_ok=True)


# In[3]:


def download_and_extract_firebrowse(cancer_type):
    """Function to download and extract clinical data for the given cancer type."""
    
    # set filenames for target cancer type
    zip_file_url = clinical_zip_files[cancer_type]
    download_file = os.path.join(cfg.msi_data_dir, 
                                 os.path.split(zip_file_url)[-1])
    download_dir = os.path.split(zip_file_url)[-1].replace('.tar.gz', '')
    print(download_file, download_dir)
    
    # retrieve compressed file from firebrowse
    urllib.request.urlretrieve(zip_file_url, download_file)
    # extract clinical data file from .tar.gz
    tar_file = tarfile.open(download_file, 'r:gz')
    tar_file.extract('gdac.broadinstitute.org_{}.Clinical_Pick_Tier1.Level_4.2016012800.0.0/All_CDEs.txt'.format(cancer_type),
                     cfg.msi_data_dir)
    tar_file.close()
    
    # move clinical data up one dir, and remove tar dir
    clinical_untar = os.path.join(cfg.msi_data_dir, download_dir,
                                  'All_CDEs.txt'.format(cancer_type))
    clinical_move_to = os.path.join(cfg.msi_data_dir, '{}_All_CDEs.txt'.format(cancer_type))
    
    # clean up untarred stuff
    os.rename(clinical_untar, clinical_move_to)
    os.remove(download_file)
    os.rmdir(os.path.join(cfg.msi_data_dir, download_dir))
    
    # return downloaded tsv filename
    return clinical_move_to
   


# In[4]:


coadread_clinical_file = download_and_extract_firebrowse('COADREAD') 
print(coadread_clinical_file)


# In[5]:


coadread_clinical_df = (
    pd.read_csv(coadread_clinical_file, sep='\t', index_col=0)
      .transpose()
)['mononucleotide_and_dinucleotide_marker_panel_analysis_status']

# rename column and uppercase TCGA identifiers to match omics datasets
coadread_clinical_df = pd.DataFrame(
    coadread_clinical_df.values,
    index=coadread_clinical_df.index.str.upper(),
    columns=['msi_status']
)

coadread_clinical_df.to_csv(
    os.path.join(cfg.msi_data_dir, 'COADREAD_msi_status.tsv'), sep='\t'
)

print(coadread_clinical_df.shape)
print(coadread_clinical_df.columns)
coadread_clinical_df.head()


# In[6]:


stad_clinical_file = download_and_extract_firebrowse('STAD') 
print(stad_clinical_file)


# In[7]:


stad_clinical_df = (
    pd.read_csv(stad_clinical_file, sep='\t', index_col=0)
      .transpose()
)['mononucleotide_and_dinucleotide_marker_panel_analysis_status']

# rename column and uppercase TCGA identifiers to match omics datasets
stad_clinical_df = pd.DataFrame(
    stad_clinical_df.values,
    index=stad_clinical_df.index.str.upper(),
    columns=['msi_status']
)

stad_clinical_df.to_csv(
    os.path.join(cfg.msi_data_dir, 'STAD_msi_status.tsv'), sep='\t'
)

print(stad_clinical_df.shape)
print(stad_clinical_df.columns)
stad_clinical_df.head()


# In[8]:


ucec_clinical_file = download_and_extract_firebrowse('UCEC') 
print(ucec_clinical_file)


# In[9]:


ucec_clinical_df = (
    pd.read_csv(ucec_clinical_file, sep='\t', index_col=0)
      .transpose()
)['mononucleotide_and_dinucleotide_marker_panel_analysis_status']

# rename column and uppercase TCGA identifiers to match omics datasets
ucec_clinical_df = pd.DataFrame(
    ucec_clinical_df.values,
    index=ucec_clinical_df.index.str.upper(),
    columns=['msi_status']
)

ucec_clinical_df.to_csv(
    os.path.join(cfg.msi_data_dir, 'UCEC_msi_status.tsv'), sep='\t'
)

print(ucec_clinical_df.shape)
print(ucec_clinical_df.columns)
ucec_clinical_df.head()

