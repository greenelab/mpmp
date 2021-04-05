## Download and process TCGA data

This directory stores scripts to download and process various data modalities from TCGA.

### Repo organization

Top-level `.ipynb` scripts should be run in order of number (i.e. `0_`, `1_`, etc) as notebooks to download and preprocess data.
Notebooks at the same level (i.e. `1A_`, `1B_`) are not dependent on one another, but may be dependent on notebooks at the previous level.

Scripts in `./nbconverted` are for code review purposes, and are not intended to be run on their own.

### Data information

The list of data used as part of this repository is listed in the [Genomic Data Commons of The National Cancer Institute](https://gdc.cancer.gov/about-data/publications/pancanatlas).
We download, process, and train our models using the datasets listed on that page.

### Methylation data preprocessing/bias correction

Optionally, we support running a preprocessing step for methylation array data that involves correcting probe intensity differences between type I and type II probes.
We use the [beta-mixture quantile normalization (BMIQ) method](https://doi.org/10.1093/bioinformatics/bts680) to do this, as implemented in the [wateRmelon R package](https://www.bioconductor.org/packages/release/bioc/html/wateRmelon.html).

We provide a conda environment file to install the R dependencies for this package here, at `bmiq_environment.yml`.
This environment can be created using the following command:
```shell
conda env create --file bmiq_environment.yml

conda activate bmiq
```

Then, follow the steps in `methylation_beta.ipynb`, and run the `run_bmiq.R` script as specified in the Jupyter notebook.
