"""
Functions for batch correction

"""
import sys

import numpy as np
import pandas as pd

def run_limma(data, batches, columns=None, coefs=None, verbose=False):
    """ Use limma to correct for batch effects.

    Arguments
    ---------
    data (np.array): a samples x features matrix to be corrected
    batches (np.array): the batch, e.g. platform, study, or experiment
                        that each sample came from, in order
    columns (np.array): bool array of columns to use, if None use all of them
    coefs (np.array): existing coefficients to use for batch correction, if
                      None fit a model on provided data to get coefs
    verbose: if True, print verbose output

    Returns
    -------
    corrected_data: samples x features dataframe of batch corrected input
    coefs: coefficients from remove_batch_effect
    """
    if verbose:
        print('Correcting for batch effects using limma...', file=sys.stderr)

    if columns is not None:
        values_to_correct = data.loc[:, columns].copy().values
    else:
        values_to_correct = data.copy().values

    corrected_values, coefs = remove_batch_effect(values_to_correct,
                                                  batches,
                                                  coefs=coefs)
    corrected_data = data.copy()

    if columns is not None:
        corrected_data.loc[:, columns] = corrected_values
    else:
        corrected_data.loc[:, :] = corrected_values

    # TODO: maybe add unit test for this?
    assert corrected_data.columns.equals(data.columns)
    assert corrected_data.shape == data.shape

    return corrected_data, coefs


def remove_batch_effect(X, batches, coefs=None):
    """Python version of limma::removeBatchEffect.

    This should duplicate the original R code here (for the case
    where there is only a single vector of batches):
    https://rdrr.io/bioc/limma/src/R/removeBatchEffect.R

    For now, batches needs to be integer indexes.

    If coefs are provided, they should be an m x p vector, where m
    is the dimension of the design matrix and p is the number of features
    in the original dataset.
    """
    from patsy.contrasts import Sum

    # use sum coding to code batches, this is what limma does
    # https://www.statsmodels.org/dev/examples/notebooks/generated/contrasts.html#Sum-(Deviation)-Coding
    # this is a bit easier/more intuitive in R, due to its built-in factor
    # type, but we can sort of emulate it here with pandas categorical data
    batches_df = pd.Series(batches, dtype='category')
    contrast = Sum().code_without_intercept(
        list(batches_df.cat.categories)
    )
    design = contrast.matrix[batches.astype(int), :]

    # if coefficients are provided, just use them to correct the provided data
    # otherwise fit the model and correct the provided data
    if coefs is None:
        from sklearn.linear_model import LinearRegression
        # X is an n x p matrix
        # batches is a n x m vector of batch indicators
        # we want to find a m x p vector of coefficients
        reg = LinearRegression().fit(design, X)
        # per sklearn documentation, for multiple targets the coef_ is
        # always an (n_targets, n_features) array (i.e. m x p)
        assert reg.coef_.shape == (X.shape[1], design.shape[1])
        coefs = reg.coef_

    return X - (design.astype(float) @ coefs.T), coefs


def limma_train_test(X_train, X_test, batches_train, batches_test, columns=None):
    """Fit batch correction model on training data, then apply to test data.

    X_train and X_test should have the same set of features, but they can have
    different numbers of samples.
    """
    X_train_adj, coefs = run_limma(X_train, batches_train, columns=columns)
    X_test_adj, _ = run_limma(X_test, batches_test, columns=columns, coefs=coefs)
    return X_train_adj, X_test_adj


def limma_ratio(X_train, X_test, batches_train, batches_test, ratio, seed):
    if ratio == 0.0:
        return X_train, X_test
    # select columns to batch correct at random
    col_subset = X_train.columns.to_series().sample(frac=ratio, random_state=seed)
    col_select = X_train.columns.isin(col_subset)
    # call limma_train_test with the selected subset
    return limma_train_test(X_train, X_test, batches_train, batches_test,
                            columns=col_select)

