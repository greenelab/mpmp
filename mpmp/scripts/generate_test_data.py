"""
Generate (or regenerate) data for regression testing of model fitting
functionality.
"""
import argparse

import numpy as np
import pandas as pd

import mpmp.test_config as tcfg
from mpmp.data_models.tcga_data_model import TCGADataModel
import mpmp.utilities.classify_utilities as cu
import mpmp.utilities.data_utilities as du

def generate_data_model(data_type, verbose=False):
    """Load data model and sample info data"""
    tcga_data = TCGADataModel(training_data=data_type,
                              test=True, verbose=verbose)
    sample_info_df = du.load_sample_info(train_data_type=data_type,
                                         verbose=verbose)
    return tcga_data, sample_info_df


def generate_stratified_test_data(tcga_data, data_type, sample_info_df, verbose=False):
    """Generate results for model fit to stratified cross-validation data"""
    for gene, classification in tcfg.stratified_gene_info:
        output_file = tcfg.test_stratified_results.format(data_type, gene)
        if verbose:
            print(data_type, gene, classification)
            print(output_file)
        tcga_data.process_data_for_gene(gene,
                                        classification,
                                        gene_dir=None,
                                        shuffle_labels=False)
        results = cu.run_cv_stratified(tcga_data,
                                       'gene',
                                       gene,
                                       data_type,
                                       sample_info_df,
                                       num_folds=4,
                                       standardize_columns=True,
                                       shuffle_labels=False)
        metrics_df = pd.concat(results['gene_metrics'])
        np.savetxt(output_file, metrics_df['auroc'].values)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # NOTE this will overwrite existing results files
    # (could eventually add a warning for this?)
    for data_type in tcfg.test_data_types:
        tcga_data, sample_info_df = generate_data_model(data_type, args.verbose)
        generate_stratified_test_data(tcga_data, data_type, sample_info_df,
                                      verbose=args.verbose)

