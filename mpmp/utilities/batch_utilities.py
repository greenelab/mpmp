"""
Functions for batch correction

"""
import sys

import numpy as np
import pandas as pd

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

    # limma expects data in features x samples format
    if correct_covariates:
        values_to_correct = data.copy().values
    else:
        values_to_correct = data.loc[:, gene_features].copy().values

    corrected_values = remove_batch_effect(values_to_correct, batches)
    corrected_data = data.copy()

    if correct_covariates:
        corrected_data.loc[:, :] = corrected_values
    else:
        corrected_data.loc[:, gene_features] = corrected_values

    # TODO: maybe add unit test for this?
    assert corrected_data.columns.equals(data.columns)
    assert corrected_data.shape == data.shape

    return corrected_data


def remove_batch_effect(X, batches):
    """Python version of limma::removeBatchEffect.

    This should duplicate the original R code here (for the case
    where there is only a single vector of batches):
    https://rdrr.io/bioc/limma/src/R/removeBatchEffect.R

    For now, batches needs to be integer indexes.
    """
    from patsy.contrasts import Sum
    from sklearn.linear_model import LinearRegression

    # use sum coding to code batches, this is what limma does
    # https://www.statsmodels.org/dev/examples/notebooks/generated/contrasts.html#Sum-(Deviation)-Coding
    # this is something that is actually easier in R, due to its
    # built-in factor type, but we can sort of emulate it here
    # with pandas categorical data
    batches_df = pd.Series(batches, dtype='category')
    contrast = Sum().code_without_intercept(
        list(batches_df.cat.categories)
    )
    design = contrast.matrix[batches.astype(int), :]

    # X is an n x p matrix
    # batches is a n x m vector of batch indicators
    # we want to find a m x p vector of coefficients
    reg = LinearRegression().fit(design, X)
    # per sklearn documentation, for multiple targets the coef_ is
    # always an (n_targets, n_features) array (i.e. m x p)
    assert reg.coef_.shape == (X.shape[1], design.shape[1])
    return X - (design.astype(float) @ reg.coef_.T)

