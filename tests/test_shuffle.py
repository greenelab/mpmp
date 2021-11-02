"""
Test cases for code to shuffle labels
"""
import pytest
import numpy as np
import pandas as pd

import mpmp.config as cfg
from mpmp.prediction.cross_validation import shuffle_by_cancer_type

@pytest.fixture(scope='module')
def expression_data():
    """Load gene expression and sample info data from files"""
    expression_df = pd.read_csv(cfg.subsampled_expression, index_col=0, sep='\t')
    expression_labels_df = pd.DataFrame({
        # generate random binary labels
        'status': np.random.randint(2, size=expression_df.shape[0]),
        # generate random "cancer types" from the letters A-E, resampled
        'DISEASE': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=expression_df.shape[0])
    }, index=expression_df.index)
    return (expression_df, expression_labels_df)


def test_shuffle_by_cancer_type(expression_data):
    """Test function to shuffle labels independently by cancer type."""

    (expression_df, expression_labels_df) = expression_data

    original_ones = expression_labels_df.groupby('DISEASE').sum()['status']
    original_labels = expression_labels_df.status.copy()

    expression_labels_df.status = shuffle_by_cancer_type(expression_labels_df,
                                                         cfg.default_seed)

    new_ones = expression_labels_df.groupby('DISEASE').sum()['status']
    new_labels = expression_labels_df.status.copy()

    # make sure labels have been shuffled
    assert (not original_labels.equals(new_labels))
    # make sure original label proportions are the same as new label proportions
    assert original_ones.equals(new_ones)

