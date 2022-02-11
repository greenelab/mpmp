"""
Script to run pan-cancer mutation classification experiments, with stratified
train and test sets, for all provided genes.
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
    NoTestSamplesError,
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
    io.add_argument('--custom_genes', nargs='*', default=None,
                    help='currently this needs to be a subset of top_50')
    io.add_argument('--gene_set', type=str,
                    choices=['top_50', 'vogelstein', '50_random', 'custom'],
                    default='top_50',
                    help='choose which gene set to use. top_50 and vogelstein are '
                         'predefined gene sets (see data_utilities), and custom allows '
                         'any gene or set of genes in TCGA, specified in --custom_genes')
    io.add_argument('--log_file', default=None,
                    help='name of file to log skipped genes to')
    io.add_argument('--results_dir', default=cfg.results_dirs['mutation'],
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
    opts.add_argument('--batch_correction', action='store_true',
                      help='if included, use limma to remove linear signal, '
                           'this is useful to determine how much non-linear signal '
                           'exists in the data')
    opts.add_argument('--bc_cancer_type', action='store_true',
                      help='if included, use limma to remove linear cancer type signal')
    opts.add_argument('--bc_train_test', action='store_true',
                      help='if included, fit BE correction model on train set, '
                           'then apply to test set')
    opts.add_argument('--debug', action='store_true',
                      help='use subset of data for fast debugging')
    opts.add_argument('--drop_target', action='store_true',
                      help='drop target gene from feature set, '
                           'currently only implemented for expression data')
    opts.add_argument('--feature_selection', choices=['f_test', 'mad', 'random'],
                      help='method to use for feature selection, only applied if '
                           '0 > num_features > total number of columns')
    opts.add_argument('--num_features', type=int, default=cfg.num_features_raw,
                      help='if included, select this number of features, using '
                           'feature selection method in feature_selection')
    opts.add_argument('--num_folds', type=int, default=4,
                      help='number of folds of cross-validation to run')
    opts.add_argument('--nonlinear', action='store_true',
                      help='use gradient-boosted classifier instead of the '
                           'default elastic net classifier')
    opts.add_argument('--only_target', action='store_true',
                      help='use only target gene + non-gene covariates, '
                           'currently only implemented for expression data')
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

    if args.gene_set == 'custom':
        if args.custom_genes is None:
            parser.error('must include --custom_genes when --gene_set=\'custom\'')
        args.gene_set = args.custom_genes
        del args.custom_genes
    elif (args.gene_set != 'custom' and args.custom_genes is not None):
        parser.error('must use option --gene_set=\'custom\' if custom genes are included')

    if args.drop_target and args.only_target:
        parser.error('drop_target and only_target are mutually exclusive')

    if (args.drop_target or args.only_target) and (args.training_data != 'expression'):
        parser.error('drop_target and only_target only implemented for expression data')

    # check that all data types in overlap_data_types are valid
    check_all_data_types(parser, args.overlap_data_types, args.debug)

    # split args into defined argument groups, since we'll use them differently
    arg_groups = du.split_argument_groups(args, parser)
    io_args, model_options = arg_groups['io'], arg_groups['model_options']

    # add some additional hyperparameters/ranges from config file to model options
    # these shouldn't be changed by the user, so they aren't added as arguments
    model_options.n_dim = None
    model_options.alphas = cfg.alphas
    model_options.l1_ratios = cfg.l1_ratios
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

    # create empty log file if it doesn't exist
    log_columns = [
        'gene',
        'training_data',
        'shuffle_labels',
        'skip_reason'
    ]

    tcga_data = TCGADataModel(seed=model_options.seed,
                              training_data=model_options.training_data,
                              overlap_data_types=model_options.overlap_data_types,
                              sample_info_df=sample_info_df,
                              verbose=io_args.verbose,
                              debug=model_options.debug)
    genes_df = tcga_data.load_gene_set(io_args.gene_set)

    # we want to run mutation prediction experiments:
    # - for true labels and shuffled labels
    #   (shuffled labels acts as our lower baseline)
    # - for all genes in the given gene set
    for shuffle_labels in (False, True):

        print('shuffle_labels: {}'.format(shuffle_labels))

        progress = tqdm(genes_df.iterrows(),
                        total=genes_df.shape[0],
                        ncols=100,
                        file=sys.stdout)

        for gene_idx, gene_series in progress:
            log_df = None
            gene = gene_series.gene
            classification = gene_series.classification
            progress.set_description('gene: {}'.format(gene))

            try:
                gene_dir = fu.make_output_dir(experiment_dir, gene)
                check_file = fu.check_output_file(gene_dir,
                                                  gene,
                                                  shuffle_labels,
                                                  model_options)
                tcga_data.process_data_for_gene(
                    gene,
                    classification,
                    gene_dir,
                    batch_correction=model_options.batch_correction,
                    bc_cancer_type=model_options.bc_cancer_type,
                    drop_target=model_options.drop_target,
                    only_target=model_options.only_target
                )
            except ResultsFileExistsError:
                # this happens if cross-validation for this gene has already been
                # run (i.e. the results file already exists)
                if io_args.verbose:
                    print('Skipping because results file exists already: gene {}'.format(
                        gene), file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [gene, model_options.training_data, shuffle_labels, 'file_exists']
                )
                fu.write_log_file(log_df, io_args.log_file)
                continue
            except KeyError:
                # this can happen if the given gene isn't in the mutation data
                print('Gene {} not found in mutation data, skipping'.format(gene),
                      file=sys.stderr)
                log_df = fu.generate_log_df(
                    log_columns,
                    [gene, model_options.training_data, shuffle_labels, 'gene_not_found']
                )
                fu.write_log_file(log_df, io_args.log_file)
                continue

            try:
                standardize_columns = (model_options.training_data in
                                       cfg.standardize_data_types)
                results = run_cv_stratified(
                    tcga_data,
                    'gene',
                    gene,
                    model_options.training_data,
                    sample_info_df,
                    model_options.num_folds,
                    predictor='classify',
                    shuffle_labels=shuffle_labels,
                    standardize_columns=standardize_columns,
                    num_features=model_options.num_features,
                    feature_selection_method=model_options.feature_selection,
                    nonlinear=model_options.nonlinear,
                    bc_train_test=model_options.bc_train_test
                )
                # only save results if no exceptions
                fu.save_results(gene_dir,
                                check_file,
                                results,
                                'gene',
                                gene,
                                shuffle_labels,
                                model_options)
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

