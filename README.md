# mpmp

**M**ultimodal **P**an-cancer **M**utation **P**rediction

**Note:** This repository is currently a work in progress, so some aspects of the code/analysis may not be fully described or documented here.

Many lines of evidence suggest that driver mutations in cancer can affect normal cellular function through a wide variety of perturbations.
These can include altered gene expression, altered DNA methylation patterns, and altered miRNA function, among others.

Thus, if our goal is to identify samples with a given driver mutation, it is plausible that different data modalities will contain different amounts of information.
For example, IDH1 mutations have been [implicated in aberrant DNA methylation](https://doi.org/10.1038/s41598-019-53262-7), and TP53 mutation has been [shown to lead to distinct and detectable gene expression changes]( https://doi.org/10.1016/j.celrep.2018.03.076).
Our goal is to develop a data-driven methodology for identifying and comparing the strength of relationships between various drivers and one or more data modalities.

More to come as the project progresses.

## Setup

We recommend using the conda environment specified in the `environment.yml` file to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate mpmp
```

## Testing pipeline

You can run the testing suite using `pytest tests` or a similar command from the base directory (i.e. where this README is). This involves both unit tests for individual functions and regression tests that ensure that changes to the code do not affect model predictions. To run only the regression tests, use `pytest tests/test_model.py`.

If you make a change to the code that is intended to affect the model output, you'll need to update the saved model output for the regression tests. This can be done by running the script at `mpmp/scripts/generate_test_data.py`. Note that this script will overwrite the existing model output files in the `tests/data` directory, so make sure this is what you want.
