"""
Script to run microsatellite instability prediction experiments.
"""
import sys
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
from mpmp.exceptions import (
    ResultsFileExistsError,
    NoTrainSamplesError,
    OneClassError,
)
from mpmp.prediction.cross_validation import run_cv_stratified
import mpmp.utilities.data_utilities as du
import mpmp.utilities.file_utilities as fu
from mpmp.utilities.tcga_utilities import check_all_data_types

def process_args():
    """Parse and format command line arguments."""

    parser = argparse.ArgumentParser()

    # argument group for parameters related to input/output
    # (e.g. filenames, logging/verbosity options, target genes)
    #
    # these don't affect the model output, and thus don't need to be saved
    # with the results of the experiment
    io = parser.add_argument_group('io',
                                   'arguments related to script input/output, '
                                   'note these will *not* be saved in metadata ')
    io.add_argument('--cancer_types', nargs='*', default=['all_cancer_types'],
                    help='cancer types to run, \'pancancer\' for a pan-cancer model '
                         'combining cancer types, default is all individual TCGA '
                         'cancer types + pan-cancer model')
    io.add_argument('--log_file', default=None,
                    help='name of file to log errors to')
    io.add_argument('--results_dir', default=cfg.results_dirs['msi'],
                    help='where to write results to')
    io.add_argument('--verbose', action='store_true')

    # argument group for parameters related to model training/evaluation
    # (e.g. model hyperparameters, preprocessing options)
    #
    # these affect the output of the model, so we want to save them in the
    # same directory as the experiment results
    opts = parser.add_argument_group('model_options',
                                     'parameters for training/evaluating model, '
                                     'these will affect output and are saved as '
                                     'experiment metadata ')
    opts.add_argument('--debug', action='store_true',
                      help='use subset of data for fast debugging')
    opts.add_argument('--num_folds', type=int, default=4,
                      help='number of folds of cross-validation to run')
    opts.add_argument('--overlap_data_types', nargs='*',
                      default=['expression'],
                      help='data types to define set of samples to use; e.g. '
                           'set of data types for a model comparison, use only '
                           'overlapping samples from these data types')
    opts.add_argument('--seed', type=int, default=cfg.default_seed)
    opts.add_argument('--subset_mad_genes', type=int, default=cfg.num_features_raw,
                      help='if included, subset gene features to this number of '
                           'features having highest mean absolute deviation')
    opts.add_argument('--training_data', type=str, default='expression',
                      choices=list(cfg.data_types.keys()),
                      help='what data type to train model on')
    # TODO use survival method for compression?
    opts.add_argument('--use_compressed', action='store_true',
                      help='use PCA compressed data rather than raw features')

    args = parser.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.tsv').resolve()

    if args.use_compressed and args.training_data not in cfg.compressed_data_types:
        parser.error(
            'data type {} does not have a compressed data source'.format(
                args.training_data)
        )

    msi_cancer_types = cfg.msi_cancer_types + ['pancancer']
    if 'all_cancer_types' in args.cancer_types:
        args.cancer_types = msi_cancer_types
    else:
        not_in_msi = set(args.cancer_types) - set(msi_cancer_types)
        if len(not_in_msi) > 0:
            parser.error('some cancer types do not have MSI labels: {}'.format(
                ' '.join(not_in_msi)))

    # check that all data types in overlap_data_types are valid
    check_all_data_types(parser, args.overlap_data_types, args.debug)

    # split args into defined argument groups, since we'll use them differently
    arg_groups = du.split_argument_groups(args, parser)
    io_args, model_options = arg_groups['io'], arg_groups['model_options']

    # always use 5000 PCs if `use_compressed==True`
    # TODO: just use n_dim option?
    model_options.n_dim = None
    if model_options.use_compressed:
        model_options.n_dim = 5000

    # add some additional hyperparameters/ranges from config file to model options
    # these shouldn't be changed by the user, so they aren't added as arguments
    model_options.max_iter = cfg.max_iter
    model_options.alphas = cfg.alphas
    model_options.l1_ratios = cfg.l1_ratios
    model_options.standardize_data_types = cfg.standardize_data_types

    return io_args, model_options

if __name__ == '__main__':

    # process command line arguments
    io_args, model_options = process_args()
    sample_info_df = du.load_sample_info(model_options.training_data,
                                         verbose=io_args.verbose)

    # create results dir and subdir for experiment if they don't exist
    experiment_dir = io_args.results_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # save model options for this experiment
    # (hyperparameters, preprocessing info, etc)
    fu.save_model_options(experiment_dir, model_options)

    print(io_args.cancer_types)
    exit()

    # create empty log file if it doesn't exist
    log_columns = [
        'training_data',
        'shuffle_labels',
        'skip_reason'
    ]
    if io_args.log_file.exists() and io_args.log_file.is_file():
        purity_log_df = pd.read_csv(io_args.log_file, sep='\t')
    else:
        purity_log_df = pd.DataFrame(columns=log_columns)
        purity_log_df.to_csv(io_args.log_file, sep='\t')

    tcga_data = TCGADataModel(seed=model_options.seed,
                              subset_mad_genes=model_options.subset_mad_genes,
                              training_data=model_options.training_data,
                              load_compressed_data=model_options.use_compressed,
                              n_dim=model_options.n_dim,
                              sample_info_df=sample_info_df,
                              verbose=io_args.verbose,
                              debug=model_options.debug)

    # we want to run purity prediction experiments for true labels and
    # shuffled labels (the latter as a lower baseline)
    progress = tqdm([False, True],
                    ncols=100,
                    file=sys.stdout)
    for shuffle_labels in progress:
        progress.set_description('shuffle labels: {}'.format(shuffle_labels))

        try:
            output_dir = fu.make_output_dir(experiment_dir, '')
            check_file = fu.check_output_file(output_dir,
                                              None,
                                              shuffle_labels,
                                              model_options,
                                              model_options.classify)
        except ResultsFileExistsError:
            # this happens if cross-validation for this gene has already been
            # run (i.e. the results file already exists)
            if io_args.verbose:
                print('Skipping because results file exists already', file=sys.stderr)
            purity_log_df = fu.generate_log_df(
                log_columns,
                [model_options.training_data, shuffle_labels, 'file_exists']
            )
            fu.write_log_file(purity_log_df, io_args.log_file)
            continue

        tcga_data.process_purity_data(experiment_dir,
                                      classify=model_options.classify)

        try:
            # for now, don't standardize methylation data
            standardize_columns = (model_options.training_data in
                                   cfg.standardize_data_types)
            results = run_cv_stratified(tcga_data,
                                        'purity',
                                        None,
                                        model_options.training_data,
                                        sample_info_df,
                                        model_options.num_folds,
                                        ('classify' if model_options.classify else 'regress'),
                                        shuffle_labels,
                                        standardize_columns,
                                        io_args.output_preds)
            # only save results if no exceptions
            fu.save_results(output_dir,
                            check_file,
                            results,
                            'purity',
                            None,
                            shuffle_labels,
                            model_options,
                            'classify' if model_options.classify else 'regression')
        except NoTrainSamplesError:
            if io_args.verbose:
                print('Skipping due to no train samples', file=sys.stderr)
            purity_log_df = fu.generate_log_df(
                log_columns,
                [model_options.training_data, shuffle_labels, 'no_train_samples']
            )
        except OneClassError:
            if io_args.verbose:
                print('Skipping due to one holdout class', file=sys.stderr)
            purity_log_df = fu.generate_log_df(
                log_columns,
                [model_options.training_data, shuffle_labels, 'one_class']
            )

        if purity_log_df is not None:
            fu.write_log_file(purity_log_df, io_args.log_file)

