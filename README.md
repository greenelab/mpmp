# mpmp: _M_ultimodal _P_an-cancer _M_utation _P_rediction

**Jake Crawford, Brock C. Christensen, Maria Chikina, Casey S. Greene**

**University of Pennsylvania, University of Colorado Anschutz Medical Campus, Dartmouth College, University of Pittsburgh**

Manuscript (in progress) is located here: https://greenelab.github.io/mpmp-manuscript/

In studies of cellular function in cancer, researchers are increasingly able to choose from many -omics assays as functional readouts.
Choosing the correct readout for a given study can be difficult, and it is not always clear which layer of cellular function is most suitable to capture the relevant signal.
Here, we consider prediction of cancer mutation status (presence or absence) from functional -omics data in the TCGA Pan-Cancer Atlas as a representative problem.

Across a collection of cancer-associated genetic alterations (point mutations and CNVs), RNA sequencing and DNA methylation were the most effective predictors of alteration state.
Surprisingly, we found that for most alterations, they were approximately equally effective predictors.
The target gene was the primary driver of performance, rather than the data type, and there was little difference between the top data types for the majority of genes.
We also found that combining data types into a single multi-omics model often provided little or no improvement in predictive ability over the best individual data type.
Based on our results, for the design of studies focused on the functional outcomes of cancer mutations, we recommend focusing on gene expression or DNA methylation as first-line readouts.

## Repository layout

`mpmp`
├──`00_download_data`: scripts to download and preprocess TCGA data
├──`01_explore_data`: exploratory data analysis (not included in paper)
├──`02_classify_mutations`: single-omics classification scripts (Figures 2-6 in paper)
│   └──`run_mutation_classification.py`: predict mutation status with raw features
│   └──`run_mutation_compressed.py`: predict mutation status with compressed features
│   └──`plot_expression_gene_sets.ipynb`: plot results of experiments comparing datasets (Figure 2)
│   └──`plot_methylation_results.ipynb`: plot results of experiments comparing expression and methylation (Figure 3)
│   └──`plot_results_n_dims.ipynb`: plot results of experiments comparing compression levels (Figure 4)
│   └──`plot_all_results.ipynb`: plot results of experiments comparing all data types (Figures 5 and 6)
├──`03_classify_cancer_type`: cancer type classification scripts (not included in paper)
├──`04_predict_tumor_purity`: tumor purity prediction scripts (not included in paper)
└──`05_classify_mutations_multimodal`: multi-omics classification scripts (Figure 7 in paper)
    └──`run_mutation_classification.py`: predict mutation status from multiple data types
    └──`plot_multimodal_results.ipynb`: plot results of multi-omics experiments (Figure 7)

## Setup

We recommend using the conda environment specified in the `environment.yml` file to run these analyses. To build and activate this environment, run:

```shell
# conda version 4.5.0
conda env create --file environment.yml

conda activate mpmp
```

## Testing pipeline

You can run the testing suite using `pytest tests` or a similar command from the base directory (i.e. where this README is). This involves both unit tests for individual functions and regression tests that ensure that changes to the code do not affect model predictions. These will also run automatically for each pull request via GitHub actions.

To run only the regression tests that check model output, use `pytest tests/test_model.py` (this may take 1-2 minutes at most).

If you make a change to the code that is intended to affect the model output, you'll need to update the saved model output for the regression tests. This can be done by running the script at `mpmp/scripts/generate_test_data.py`. Note that this script will overwrite the existing model output files in the `tests/data` directory, so make sure this is what you want.
