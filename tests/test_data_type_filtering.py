"""
Test cases for cross-validation code in data_utilities.py
"""
import pytest
import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.tcga_utilities as tu

@pytest.fixture(scope='module')
def expression_data():
    """Load gene expression and sample info data from files"""
    expression_df = pd.read_csv(cfg.subsampled_expression, index_col=0, sep='\t')
    expression_labels_df = pd.DataFrame({
        'status': np.random.randint(2, size=expression_df.shape[0])
    }, index=expression_df.index)
    return (expression_df, expression_labels_df)


@pytest.fixture(scope='module')
def methylation_data():
    methylation_df = pd.read_csv(cfg.subsampled_methylation, index_col=0, sep='\t')
    methylation_labels_df = pd.DataFrame({
        'status': np.random.randint(2, size=methylation_df.shape[0])
    }, index=methylation_df.index)
    return (methylation_df, methylation_labels_df)


def test_filtering_expression(expression_data, methylation_data):
    """Test filtering function for expression data."""

    (expression_df,
     expression_labels_df) = expression_data
    (methylation_df, _) = methylation_data

    X_exp_df, y_exp_df = tu.filter_to_cross_data_samples(expression_df,
                                                         expression_labels_df,
                                                         use_subsampled=True)

    # check that the indexes are the same
    assert X_exp_df.index.equals(y_exp_df.index)

    # check that the indexes are subsets of both expression and methylation
    assert np.count_nonzero(X_exp_df.index.isin(expression_df.index)) == len(X_exp_df.index)
    assert np.count_nonzero(X_exp_df.index.isin(methylation_df.index)) == len(X_exp_df.index)

    # check that there are no duplicate indexes
    assert np.count_nonzero(X_exp_df.duplicated()) == 0

    # check that columns haven't been altered
    assert X_exp_df.shape[1] == expression_df.shape[1]


def test_filtering_methylation(expression_data, methylation_data):
    """Test filtering function for methylation data."""

    (expression_df, _) = expression_data
    (methylation_df,
     methylation_labels_df) = methylation_data

    X_me_df, y_me_df = tu.filter_to_cross_data_samples(methylation_df,
                                                       methylation_labels_df,
                                                       use_subsampled=True)
    # check that the indexes are the same
    assert X_me_df.index.equals(y_me_df.index)

    # check that the indexes are subsets of both expression and methylation
    assert np.count_nonzero(X_me_df.index.isin(expression_df.index)) == len(X_me_df.index)
    assert np.count_nonzero(X_me_df.index.isin(methylation_df.index)) == len(X_me_df.index)

    # check that there are no duplicate indexes
    assert np.count_nonzero(X_me_df.duplicated()) == 0

    # check that columns haven't been altered
    assert X_me_df.shape[1] == methylation_df.shape[1]

