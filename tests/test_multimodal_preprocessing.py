"""
Test cases for multimodal preprocessing in tcga_utilities.py
"""
import string
import itertools as it

import pytest
import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.tcga_utilities as tu

@pytest.fixture(scope='module')
def raw_data():
    """Generate a raw dataset"""
    cols = list(string.ascii_lowercase)[:11]
    X_train_raw_df = pd.DataFrame(np.random.uniform(size=(20, 11)), columns=cols)
    X_test_raw_df = pd.DataFrame(np.random.uniform(size=(10, 11)), columns=cols)
    data_types = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                           cfg.NONGENE_FEATURE, cfg.NONGENE_FEATURE])
    gene_features = np.array([True] * 9 + [False] * 2)
    return (X_train_raw_df,
            X_test_raw_df,
            gene_features,
            data_types)

@pytest.mark.parametrize('standardize_columns', [[False, False, False],
                                                 [True, True, True],
                                                 [True, False, True]])
@pytest.mark.parametrize('subset_mad_genes', [-1, 1, 2])
def test_preprocessing(raw_data, standardize_columns, subset_mad_genes):
    (X_train_raw_df,
     X_test_raw_df,
     gene_features,
     data_types) = raw_data
    X_train_df, X_test_df = tu.preprocess_multi_data(X_train_raw_df,
                                                     X_test_raw_df,
                                                     gene_features,
                                                     data_types,
                                                     standardize_columns,
                                                     subset_mad_genes)
    # make sure no samples were lost
    assert X_train_df.shape[0] == X_train_raw_df.shape[0]
    assert X_test_df.shape[0] == X_test_raw_df.shape[0]
    # make sure no NaN values
    assert X_train_df.isna().sum().sum() == 0
    # make sure no non-gene features were lost, and make sure that
    # each data type has the correct number of subset columns
    if subset_mad_genes == -1:
        # if no subsetting by MAD, number of features shouldn't change
        assert (X_train_df.shape[1] == X_train_raw_df.shape[1])
    else:
        # if we do subset by MAD:
        # total number of features = (number of data types * number of features to
        # subset to) + number of non-gene features
        assert (X_train_df.shape[1] ==
                  (subset_mad_genes * (np.unique(data_types).shape[0] - 1)) +
                  (np.count_nonzero(data_types == cfg.NONGENE_FEATURE)))
    # make sure standardized columns were actually standardized
    for ix, std_col in enumerate(standardize_columns):
        if subset_mad_genes == -1:
            data_types_filtered = data_types
        else:
            # here we have to reconstruct the data types of each column, since
            # we filtered columns by MAD
            #
            # we can do that by adding each index in order * the number of
            # top MAD columns we took, then adding the non-gene features to the end
            data_types_filtered = sum(
                [[ix] * subset_mad_genes
                    for ix in range(np.unique(data_types).shape[0] - 1)],
                []
            )
            data_types_filtered += [cfg.NONGENE_FEATURE] * np.count_nonzero(
                    data_types == cfg.NONGENE_FEATURE)
            data_types_filtered = np.array(data_types_filtered)
        valid_cols = (data_types_filtered == ix)
        if std_col:
            # if a column is standardized, it should be ~standard normal, with
            # values above and below 0
            assert X_train_df.loc[:, valid_cols].values.flatten().min() < 0
            assert X_train_df.loc[:, valid_cols].values.flatten().max() > 0
        else:
            # if a column is not standardized, we sampled from a uniform (0, 1)
            # so it should only have values above 0
            assert X_train_df.loc[:, valid_cols].values.flatten().min() > 0
            assert X_train_df.loc[:, valid_cols].values.flatten().max() > 0


