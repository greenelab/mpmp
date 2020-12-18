"""
Script to run one-vs-rest cancer type classification, with stratified train and
test sets, for all provided TCGA cancer types.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
from mpmp.exceptions import (
    ResultsFileExistsError,
    NoTrainSamplesError,
    NoTestSamplesError,
)
from mpmp.utilities.classify_utilities import run_cv_cancer_type
import mpmp.utilities.data_utilities as du
import mpmp.utilities.file_utilities as fu

def process_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cancer_types', nargs='*', default=None,
                   help='cancer types to predict, if not included predict '
                        'all cancer types in TCGA')
    p.add_argument('--debug', action='store_true',
                   help='use subset of data for fast debugging')
    p.add_argument('--log_file', default=None,
                   help='name of file to log skipped cancer types to')
    p.add_argument('--num_folds', type=int, default=4,
                   help='number of folds of cross-validation to run')
    p.add_argument('--results_dir', default=cfg.results_dir,
                   help='where to write results to')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                   help='if included, subset gene features to this number of '
                        'features having highest mean absolute deviation')
    p.add_argument('--training_data', type=str, default='expression',
                   choices=['expression', 'methylation'],
                   help='what data type to train model on')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.tsv').resolve()

    # check that all provided cancer types are valid TCGA acronyms
    sample_info_df = du.load_sample_info(args.verbose)
    tcga_cancer_types = list(np.unique(sample_info_df.cancer_type))

    if args.cancer_types is None:
        args.cancer_types = tcga_cancer_types
    else:
        not_in_tcga = set(args.cancer_types) - set(tcga_cancer_types)
        if len(not_in_tcga) > 0:
            p.error('some cancer types not present in TCGA: {}'.format(
                ' '.join(not_in_tcga)))

    return args, sample_info_df


if __name__ == '__main__':

    # process command line arguments
    args, sample_info_df = process_args()

    # create results dir if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # create empty log file if it doesn't exist
    log_columns = [
        'cancer_type',
        'training_data',
        'shuffle_labels',
        'skip_reason'
    ]
    if args.log_file.exists() and args.log_file.is_file():
        log_df = pd.read_csv(args.log_file, sep='\t')
    else:
        log_df = pd.DataFrame(columns=log_columns)
        log_df.to_csv(args.log_file, sep='\t')

    tcga_data = TCGADataModel(seed=args.seed,
                              subset_mad_genes=args.subset_mad_genes,
                              training_data=args.training_data,
                              verbose=args.verbose,
                              debug=args.debug)

    # we want to run cancer type classification experiments:
    # - for true labels and shuffled labels
    #   (shuffled labels acts as our lower baseline)
    # - for all cancer types in the given list of TCGA cancers
    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        progress = tqdm(args.cancer_types,
                        total=len(args.cancer_types),
                        ncols=100,
                        file=sys.stdout)

        for cancer_type in progress:
            cancer_type_log_df = None
            progress.set_description('cancer type: {}'.format(cancer_type))

            try:
                cancer_type_dir = fu.make_output_dir(args.results_dir,
                                                     cancer_type,
                                                     'cancer_type')
                check_file = fu.check_output_file(cancer_type_dir,
                                                  cancer_type,
                                                  args.training_data,
                                                  shuffle_labels,
                                                  args.seed)
                tcga_data.process_data_for_cancer_type(cancer_type,
                                                       cancer_type_dir,
                                                       shuffle_labels=shuffle_labels)
            except ResultsFileExistsError:
                # this happens if cross-validation for this cancer type has
                # already been run (i.e. the results file already exists)
                if args.verbose:
                    print('Skipping because results file exists already: '
                          'cancer type {}'.format(cancer_type), file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [cancer_type, args.training_data, shuffle_labels, 'file_exists']
                )
                fu.write_log_file(cancer_type_log_df, args.log_file)
                continue

            try:
                # for now, don't standardize methylation data
                standardize_columns = (args.training_data in ['expression'])
                results = run_cv_cancer_type(tcga_data,
                                             cancer_type,
                                             args.training_data,
                                             sample_info_df,
                                             args.num_folds,
                                             shuffle_labels,
                                             standardize_columns)
            except NoTestSamplesError:
                if args.verbose:
                    print('Skipping due to no test samples: cancer type '
                          '{}'.format(cancer_type), file=sys.stderr)
                cancer_type_log_df = fu.generate_log_df(
                    log_columns,
                    [cancer_type, args.training_data, shuffle_labels, 'no_test_samples']
                )
            else:
                # only save results if no exceptions
                fu.save_results(cancer_type_dir,
                                check_file,
                                results,
                                'cancer_type',
                                cancer_type,
                                args.training_data,
                                shuffle_labels,
                                args.seed)

            if cancer_type_log_df is not None:
                fu.write_log_file(cancer_type_log_df, args.log_file)

