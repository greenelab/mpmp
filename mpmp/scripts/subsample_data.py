"""
Generate (or regenerate) subsampled datasets for debugging changes.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du

def subsample_stratified(data_df, sample_info_df):
    """Subsample from data_df, stratified by id in sample_info_df."""

    # reindex sample info to samples found in data
    stratify_samples_id = (
        sample_info_df.reindex(data_df.index)
                      .dropna()
                      .loc[:, ['id_for_stratification']]
    )
    # reindex data to samples found in sample info
    filtered_data_df = (
        data_df.reindex(sample_info_df.index)
               .dropna()
    )

    # pool low count ids, train_test_split requires this
    stratify_counts = stratify_samples_id.value_counts().to_dict()
    stratify_samples_id['sample_count'] = (
        stratify_samples_id.id_for_stratification.replace(stratify_counts)
    )

    # make sure indexes are in same order, they should be
    assert np.array_equal(stratify_samples_id.index, filtered_data_df.index)

    # then subsample from data, stratified by cancer type/subtype
    _, subsample_df = train_test_split(filtered_data_df,
                                       test_size=0.1,
                                       random_state=cfg.default_seed,
                                       stratify=stratify_samples_id.sample_count)

    # also subsample columns/features, this speeds up testing considerably
    subsample_df = subsample_df.sample(n=100, axis=1,
                                       random_state=cfg.default_seed)

    return subsample_df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str,
                   choices=['expression', 'me_27k', 'all'],
                   default='all')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    cfg.subsampled_data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ['expression', 'all']:
        sample_info_df = du.load_sample_info(train_data_type='expression',
                                             verbose=args.verbose)
        rnaseq_df = du.load_raw_data(train_data_type='expression',
                                     verbose=args.verbose)
        if args.verbose:
            print('Generating subsampled expression data...', end='')
        subsample_df = subsample_stratified(rnaseq_df, sample_info_df)
        subsample_df.to_csv(cfg.subsampled_expression, sep='\t',
                            compression='gzip', float_format='%.3g')
        if args.verbose:
            print('done')

    if args.dataset in ['me_27k', 'all']:
        sample_info_df = du.load_sample_info(train_data_type='me_27k',
                                             verbose=args.verbose)
        methylation_df = du.load_raw_data(train_data_type='me_27k',
                                          verbose=args.verbose)
        if args.verbose:
            print('Generating subsampled methylation data...', end='')
        subsample_df = subsample_stratified(methylation_df, sample_info_df)
        subsample_df.to_csv(cfg.subsampled_methylation, sep='\t',
                            compression='gzip', float_format='%.3g')
        if args.verbose:
            print('done')


