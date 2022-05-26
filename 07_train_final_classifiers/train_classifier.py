"""
Script to train a single pan-cancer classifier, on the whole TCGA dataset,
for a given gene and data type.
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

import mpmp.config as cfg
from mpmp.data_models.tcga_data_model import TCGADataModel
from mpmp.exceptions import (
    ResultsFileExistsError,
    NoTrainSamplesError,
    NoTestSamplesError,
    OneClassError,
)
from mpmp.prediction.cross_validation import train_all_data
import mpmp.utilities.data_utilities as du
import mpmp.utilities.file_utilities as fu
import mpmp.utilities.param_results_utilities as pru
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
    io.add_argument('--gene', type=str, default='TP53',
                    help='choose which gene to train a classifier for, must be in cancer '
                         'dataset')
    io.add_argument('--log_file', default=None,
                    help='name of file to log skipped genes to')
    io.add_argument('--params_dir', required=True,
                    help='where to look for hyperparameter search results, in order '
                         'to find the set of hyperparameters to use')
    io.add_argument('--results_dir', default=cfg.results_dirs['mutation'],
                    help='where to write results to')
    io.add_argument('--save_model', action='store_true',
                    help='if true, serialize model and save to results dir')
    io.add_argument('--verbose', action='store_true')

    # argument group for parameters related to model training/evaluation
    # (e.g. model hyperparameters, preprocessing options)
    #
    # these affect the output of the model, so we want to save them in the
    # same directory as the experiment results
    opts = parser.add_argument_group('model_options',
                                     'parameters for training model, '
                                     'these will affect output and are saved as '
                                     'experiment metadata ')
    opts.add_argument('--debug', action='store_true',
                      help='use subset of data for fast debugging')
    opts.add_argument('--feature_selection',
                      choices=['f_test', 'mad', 'random'],
                      default='mad',
                      help='method to use for feature selection, only applied if '
                           '0 > num_features > total number of columns')
    opts.add_argument('--model', choices=cfg.model_choices, default='elasticnet',
                      help='what type of model to use for classification, defaults '
                           'to logistic regression with elastic net regularization')
    opts.add_argument('--n_dim', default=None,
                      help='number of compressed dimensions to use, if None '
                           'use raw features')
    opts.add_argument('--num_features', type=int, default=cfg.num_features_raw,
                      help='if included, select this number of features, using '
                           'feature selection method in feature_selection')
    opts.add_argument('--overlap_data_types', nargs='*',
                      default=['expression'],
                      help='data types to define set of samples to use; e.g. '
                           'set of data types for a model comparison, use only '
                           'overlapping samples from these data types')
    opts.add_argument('--seed', type=int, default=cfg.default_seed)
    opts.add_argument('--training_data', type=str, default='expression',
                      choices=list(cfg.data_types.keys()),
                      help='what data type to train model on')

    args = parser.parse_args()

    args.results_dir = Path(args.results_dir).resolve()

    if args.log_file is None:
        args.log_file = Path(args.results_dir, 'log_skipped.tsv').resolve()

    # check that all data types in overlap_data_types are valid
    check_all_data_types(parser, args.overlap_data_types, args.debug)

    # split args into defined argument groups, since we'll use them differently
    arg_groups = du.split_argument_groups(args, parser)
    io_args, model_options = arg_groups['io'], arg_groups['model_options']

    # add some additional hyperparameters/ranges from config file to model options
    # these shouldn't be changed by the user, so they aren't added as arguments
    if model_options.n_dim is not None:
        model_options.n_dim = int(model_options.n_dim)
    model_options.standardize_data_types = cfg.standardize_data_types
    model_options.shuffle_by_cancer_type = cfg.shuffle_by_cancer_type

    return io_args, model_options


if __name__ == '__main__':

    # process command line arguments
    io_args, model_options = process_args()
    sample_info_df = du.load_sample_info(model_options.training_data,
                                         verbose=io_args.verbose)

    # create results dir and subdir for experiment if they don't exist
    experiment_dir = Path(io_args.results_dir, 'gene').resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # save model options for this experiment
    # (hyperparameters, preprocessing info, etc)
    fu.save_model_options(experiment_dir, model_options)

    # get hyperparameters to use
    if io_args.verbose:
        print('Selecting best params from results directory', file=sys.stderr)
    best_params = pru.get_best_params(io_args.params_dir, io_args.gene)
    params_to_use = pru.sample_from_param_results(best_params,
                                                  model_options.seed,
                                                  model_options.training_data)
    if io_args.verbose:
        print('Params:', params_to_use, file=sys.stderr)

    # create empty log file if it doesn't exist
    log_columns = [
        'gene',
        'training_data',
        'skip_reason'
    ]

    tcga_data = TCGADataModel(seed=model_options.seed,
                              training_data=model_options.training_data,
                              overlap_data_types=model_options.overlap_data_types,
                              load_compressed_data=(model_options.n_dim is not None),
                              standardize_input=(model_options.n_dim is not None),
                              n_dim=model_options.n_dim,
                              sample_info_df=sample_info_df,
                              verbose=io_args.verbose,
                              debug=model_options.debug)

    genes_df = tcga_data.load_gene_set('merged')
    gene_series = genes_df[genes_df.gene == io_args.gene]
    gene = gene_series.gene.values[0]
    classification = gene_series.classification.values[0]

    log_df = None

    # load gene labels and related info
    try:
        gene_dir = fu.make_output_dir(experiment_dir, gene)
        check_file = fu.check_output_file(gene_dir,
                                          gene,
                                          False,
                                          model_options)
        tcga_data.process_data_for_gene(
            gene,
            classification,
            gene_dir
        )
    except ResultsFileExistsError:
        # this happens if cross-validation for this gene has already been
        # run (i.e. the results file already exists)
        if io_args.verbose:
            print('Skipping because results file exists already: gene {}'.format(
                gene), file=sys.stderr)
        log_df = fu.generate_log_df(
            log_columns,
            [gene, model_options.training_data, 'file_exists']
        )
        fu.write_log_file(log_df, io_args.log_file)
        exit()
    except KeyError:
        # this can happen if the given gene isn't in the mutation data
        print('Gene {} not found in mutation data, skipping'.format(gene),
              file=sys.stderr)
        log_df = fu.generate_log_df(
            log_columns,
            [gene, model_options.training_data, 'gene_not_found']
        )
        fu.write_log_file(log_df, io_args.log_file)
        exit()

    # train model
    try:
        # don't standardize if using compressed data, or if
        # training data is not in config data types to standardize
        standardize_columns = ((model_options.n_dim is None) and
                               (model_options.training_data in cfg.standardize_data_types))
        model_fit, results = train_all_data(
            tcga_data,
            'gene',
            gene,
            model_options.training_data,
            sample_info_df,
            params_to_use[model_options.training_data],
            standardize_columns=standardize_columns,
            num_features=model_options.num_features,
            feature_selection_method=model_options.feature_selection,
        )
        fu.save_results(gene_dir,
                        check_file,
                        results,
                        'gene',
                        gene,
                        False,
                        model_options)
        fu.save_best_params(
            gene_dir,
            params_to_use[model_options.training_data],
            gene,
            model_options
        )
        if io_args.save_model:
            fu.save_model(gene_dir, model_fit, gene, model_options)
    except NoTrainSamplesError:
        if io_args.verbose:
            print('Skipping due to no train samples: gene {}'.format(
                gene), file=sys.stderr)
        log_df = fu.generate_log_df(
            log_columns,
            [gene, model_options.training_data, shuffle_labels, 'no_train_samples']
        )
    except OneClassError:
        if io_args.verbose:
            print('Skipping due to one holdout class: gene {}'.format(
                gene), file=sys.stderr)
        log_df = fu.generate_log_df(
            log_columns,
            [gene, model_options.training_data, shuffle_labels, 'one_class']
        )

    if log_df is not None:
        fu.write_log_file(log_df, io_args.log_file)


