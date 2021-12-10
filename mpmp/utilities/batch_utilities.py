"""
Functions for batch correction

"""
import sys

import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def run_limma(data, batches, gene_features, correct_covariates=True, verbose=False):
    """ Use limma to correct for batch effects.

    Adapted from:
    https://github.com/greenelab/saged/blob/master/saged/utils.py

    Arguments
    ---------
    data (np.array): a samples x features matrix to be corrected
    batches (np.array): the batch, e.g. platform, study, or experiment
                        that each sample came from, in order
    gene_features (list): a bool list of gene/not gene features
    correct_covariates: if True, apply batch correction to non-gene covariates
    verbose: if True, print verbose output

    Returns
    -------
    corrected_data: samples x features dataframe of batch corrected input
    """
    if verbose:
        print('Correcting for batch effects using limma...', file=sys.stderr)

    limma = importr('limma')

    # limma expects data in features x samples format
    if correct_covariates:
        values_to_correct = data.copy().values.T
    else:
        values_to_correct = data.loc[:, gene_features].copy().values.T

    pandas2ri.activate()

    corrected_values = limma.removeBatchEffect(values_to_correct, batches)

    corrected_data = data.copy()

    if correct_covariates:
        corrected_data.loc[:, :] = corrected_values.T
    else:
        corrected_data.loc[:, gene_features] = corrected_values.T

    # TODO: maybe add unit test for this?
    assert corrected_data.columns.equals(data.columns)
    assert corrected_data.shape == data.shape

    return corrected_data
