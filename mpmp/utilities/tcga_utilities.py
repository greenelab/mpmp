"""
Functions for preprocessing TCGA expression data and mutation status labels.

Most functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_y_matrix(y_mutation,
                     y_copy,
                     include_copy,
                     gene,
                     sample_freeze,
                     mutation_burden,
                     filter_count,
                     filter_prop,
                     output_directory,
                     hyper_filter=5,
                     test=False):
    """
    Combine copy number and mutation data and filter cancer-types to build y matrix

    Arguments
    ---------
    y_mutation: Pandas DataFrame of mutation status
    y_copy: Pandas DataFrame of copy number status
    include_copy: boolean if the copy number data should be included in status calc
    gene: string indicating gene of interest (used for writing proportion file)
    sample_freeze: pandas dataframe storing which samples to use
    mutation_burden: pandas dataframe storing log10 mutation counts
    filter_count: the number of positives or negatives required per cancer-type
    filter_prop: the proportion of positives or negatives required per cancer-type
    output_directory: the name of the directory to store the gene summary
    hyper_filter: the number of std dev above log10 mutation burden to filter
    test: if true, don't write filtering info to disk

    Returns
    -------
    Write file of cancer-type filtering to disk and output a processed y vector
    """
    if include_copy:
        y_df = y_copy + y_mutation
    else:
        y_df = y_mutation

    y_df.loc[y_df > 1] = 1
    y_df = pd.DataFrame(y_df)
    y_df.columns = ["status"]

    y_df = (
        y_df.merge(
            sample_freeze, how="left", left_index=True, right_on="SAMPLE_BARCODE"
        )
        .set_index("SAMPLE_BARCODE")
        .merge(mutation_burden, left_index=True, right_index=True)
    )

    # Get statistics per gene and disease
    disease_counts_df = pd.DataFrame(y_df.groupby("DISEASE").sum()["status"])

    disease_proportion_df = disease_counts_df.divide(
        y_df["DISEASE"].value_counts(sort=False).sort_index(), axis=0
    )

    # Filter diseases with low counts or proportions for classification balance
    filter_disease_df = (disease_counts_df > filter_count) & (
        disease_proportion_df > filter_prop
    )
    filter_disease_df.columns = ["disease_included"]

    disease_stats_df = disease_counts_df.merge(
        disease_proportion_df,
        left_index=True,
        right_index=True,
        suffixes=("_count", "_proportion"),
    ).merge(filter_disease_df, left_index=True, right_index=True)

    if not test:
        filter_file = "{}_filtered_cancertypes.tsv".format(gene)
        filter_file = os.path.join(output_directory, filter_file)
        disease_stats_df.to_csv(filter_file, sep="\t")

    # Filter
    use_diseases = disease_stats_df.query("disease_included").index.tolist()
    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :].query("DISEASE in @use_diseases")

    return y_df


def process_y_matrix_cancertype(
    acronym, sample_freeze, mutation_burden, hyper_filter=5
):
    """Build a y vector based on cancer-type membership.

    Arguments
    ---------
    acronym (str): the TCGA cancer-type barcode
    sample_freeze (pd.DataFrame): stores TCGA barcodes and cancer-types
    mutation_burden (pd.DataFrame): log10 mutation count per sample
                                    (this gets added as covariate)
    hyper_filter (float): the number of std dev above log10 mutation burden
                          to filter

    Returns
    -------
    y_df: 0/1 status DataFrame for the given cancer type
    count_df: status count dataframe
    """
    y_df = sample_freeze.assign(status=0)
    y_df.loc[y_df.DISEASE == acronym, "status"] = 1

    y_df = y_df.set_index("SAMPLE_BARCODE").merge(
        mutation_burden, left_index=True, right_index=True
    )

    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :]

    count_df = pd.DataFrame(y_df.status.value_counts()).reset_index()
    count_df.columns = ["status", acronym]

    return y_df, count_df


def align_matrices(x_file_or_df,
                   y,
                   add_cancertype_covariate=True,
                   add_mutation_covariate=True):
    """
    Process the x matrix for the given input file and align x and y together

    Arguments
    ---------
    x_file_or_df: string location of the x matrix or matrix df itself
    y: pandas DataFrame storing status of corresponding samples
    add_cancertype_covariate: if true, add one-hot encoded cancer type as a covariate
    add_mutation_covariate: if true, add log10(mutation burden) as a covariate

    Returns
    -------
    use_samples: the samples used to subset
    rnaseq_df: processed X matrix
    y_df: processed y matrix
    gene_features: real-valued gene features, to be standardized later
    """
    try:
        x_df = pd.read_csv(x_file_or_df, index_col=0, sep='\t')
    except:
        x_df = x_file_or_df

    # select samples to use, assuming y has already been filtered by cancer type
    use_samples = y.index.intersection(x_df.index)
    x_df = x_df.reindex(use_samples)
    y = y.reindex(use_samples)

    # add features to X matrix if necessary
    gene_features = np.ones(x_df.shape[1]).astype('bool')

    if add_cancertype_covariate:
        # add one-hot covariate for cancer type
        covariate_df = pd.get_dummies(y.DISEASE)
        x_df = x_df.merge(covariate_df, left_index=True, right_index=True)

    if add_mutation_covariate:
        # add covariate for mutation burden
        mutation_covariate_df = pd.DataFrame(y.loc[:, "log10_mut"], index=y.index)
        x_df = x_df.merge(mutation_covariate_df, left_index=True, right_index=True)

    num_added_features = x_df.shape[1] - gene_features.shape[0]
    if num_added_features > 0:
        gene_features = np.concatenate(
            (gene_features, np.zeros(num_added_features).astype('bool'))
        )

    return use_samples, x_df, y, gene_features
