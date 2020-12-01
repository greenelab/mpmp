# mpmp

**M**ultimodal **P**an-cancer **M**utation **P**rediction

**Note:** This repository is currently a work in progress, so some aspects of
the code/analysis may not be fully described or documented here.

Many lines of evidence suggest that driver mutations in cancer can operate
through a wide variety of perturbations of normal cellular function. These can
include altered gene expression, altered DNA methylation patterns and altered
miRNA function, among others.

Thus, if our goal is to identify samples with a given driver mutation, it is
plausible that different data modalities will contain different amounts of
information. For example, IDH1 mutations have been [implicated in aberrant DNA
methylation](https://doi.org/10.1038/s41598-019-53262-7), and TP53 mutation
has been [shown to lead to distinct and detectable gene expression changes](
https://doi.org/10.1016/j.celrep.2018.03.076). More to come.

## Setup

We recommend using the conda environment specified in the `environment.yml` file
to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate mpmp
```


