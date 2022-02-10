"""
Test cases for drop_target/only_target functions in tcga_utilities.py
"""
import string

import pytest
import numpy as np
import pandas as pd

import mpmp.utilities.tcga_utilities as tu

@pytest.fixture(scope='module')
def raw_data():
    """Generate a raw dataset"""
    cols = list(string.ascii_lowercase)[:11]
    X_train_raw_df = pd.DataFrame(np.random.uniform(size=(20, 11)), columns=cols)
    X_test_raw_df = pd.DataFrame(np.random.uniform(size=(10, 11)), columns=cols)
    gene_features = np.array([True] * 9 + [False] * 2)
    return (X_train_raw_df,
            X_test_raw_df,
            gene_features)

@pytest.mark.parametrize('target_feature', ['a', 'c', 'e'])
def test_drop_target(raw_data, target_feature):
    (X_train_raw_df,
     X_test_raw_df,
     gene_features) = raw_data
    # TODO: make sure gene_features filtering is idempotent
    X_train_df, gene_features_train = tu.drop_target_from_data(X_train_raw_df,
                                                               target_feature,
                                                               gene_features)
    X_test_df, gene_features_test = tu.drop_target_from_data(X_test_raw_df,
                                                             target_feature,
                                                             gene_features)

    # gene features preprocessing should work the same, whether we pass
    # in train or test data (they have the same features)
    assert np.array_equal(gene_features_train, gene_features_test)

    # sample set should be the same
    assert X_train_raw_df.index.equals(X_train_df.index)
    assert X_test_raw_df.index.equals(X_test_df.index)

    # columns should be the same
    assert X_train_df.columns.equals(X_test_df.columns)

    # target feature should not be present in columns
    assert not (target_feature in X_train_df.columns)
    assert not (target_feature in X_test_df.columns)

    # other features should all be present
    assert (X_train_df.columns[gene_features_train].equals(
            X_train_raw_df.columns[gene_features].drop(target_feature)))
    assert (X_test_df.columns[gene_features_test].equals(
            X_test_raw_df.columns[gene_features].drop(target_feature)))


