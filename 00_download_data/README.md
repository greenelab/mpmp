## Download and process TCGA data

This directory stores scripts to download and process various data modalities
from TCGA.

### Repo organization

Top-level `.ipynb` scripts should be run in order of number (i.e. `0_`, `1_`,
etc) as notebooks to download and preprocess data. Notebooks at the same level
(i.e. `1A_`, `1B_`) are not dependent on one another, but may be dependent on
notebooks at the previous level.

Scripts in `./nbconverted` are for code review purposes, and are not intended
to be run on their own.

### Data information

The list of data used as part of this repository is listed in the
[Genomic Data Commons of The National Cancer Institute](
https://gdc.cancer.gov/about-data/publications/pancanatlas).
We download, process, and train our models using the `RNA (Final)`
and `DNA Methylation (Merged 27K+450K Only)` data listed there.
