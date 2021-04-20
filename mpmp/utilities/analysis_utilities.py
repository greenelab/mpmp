import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def load_stratified_prediction_results(results_dir, experiment_descriptor):
    """Load results of stratified prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes or cancer types
    experiment_descriptor (str): string describing this experiment, can be
                                 useful to segment analyses involving multiple
                                 experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for identifier in results_dir.iterdir():
        identifier_dir = Path(results_dir, identifier)
        if identifier_dir.is_file(): continue
        for results_file in identifier_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.stem)
            # skip compressed files here, use load_compressed* functions
            # to load that data separately
            if check_compressed_file(results_filename): continue
            if ('classify' not in results_filename or
                'metrics' not in results_filename): continue
            if results_filename[0] == '.': continue
            id_results_df = pd.read_csv(results_file, sep='\t')
            id_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, id_results_df))
    return results_df


def load_compressed_prediction_results(results_dir,
                                       experiment_descriptor,
                                       old_filenames=False):
    """Load results of compressed prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes or cancer types
    experiment_descriptor (str): string describing this experiment, can be
                                 useful to segment analyses involving multiple
                                 experiments or results sets
    old_filenames (bool): use old filename format

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for identifier in results_dir.iterdir():
        identifier_dir = Path(results_dir, identifier)
        if identifier_dir.is_file(): continue
        for results_file in identifier_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.stem)
            if not check_compressed_file(results_filename): continue
            if ('classify' not in results_filename or
                'metrics' not in results_filename): continue
            if results_filename[0] == '.': continue
            if old_filenames:
                try:
                    n_dims = int(results_filename.split('_')[-3].replace('n', ''))
                except ValueError:
                    n_dims = int(results_filename.split('_')[-2].replace('n', ''))
            else:
                n_dims = int(results_filename.split('_')[-2].replace('n', ''))
            id_results_df = pd.read_csv(results_file, sep='\t')
            id_results_df['n_dims'] = n_dims
            id_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, id_results_df))
    return results_df


def load_purity_results(results_dir, classify=True):
    """Load results of tumor purity experiments.

    Arguments
    ---------
    results_dir (str): directory containing results files

    Returns
    -------
    results_df (pd.DataFrame): results of prediction experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for results_file in results_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if classify and ('classify' not in results_filename
                         or 'metrics' not in results_filename): continue
        if not classify and ('regress' not in results_filename
                             or 'metrics' not in results_filename): continue
        if results_filename[0] == '.': continue
        id_results_df = pd.read_csv(results_file, sep='\t')
        if check_compressed_file(results_filename):
            id_results_df.training_data += '_compressed'
        results_df = pd.concat((results_df, id_results_df))
    return results_df


def load_purity_by_cancer_type(results_dir, sample_info_df, classify=True):
    """Load results of tumor purity prediction, grouped by cancer type.

    Assumes labels are binarized into above/below median.

    Arguments
    ---------
    results_dir (str): directory containing results files
    sample_info_df (pd.DataFrame): contains cancer type info for samples
    classify (bool): look for classification results if true

    Returns
    -------
    results_df (pd.DataFrame): results of prediction experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for results_file in results_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if classify and ('classify' not in results_filename
                         or 'preds' not in results_filename): continue
        if not classify and ('regress' not in results_filename
                             or 'preds' not in results_filename): continue
        if results_filename[0] == '.': continue
        if check_compressed_file(results_filename):
            training_data = '_'.join(results_filename.split('_')[:-5])
            training_data += '_compressed'
            signal = results_filename.split('_')[-5]
            seed = int(results_filename.split('_')[-3].replace('s', ''))
        else:
            training_data = '_'.join(results_filename.split('_')[:-4])
            signal = results_filename.split('_')[-4]
            seed = int(results_filename.split('_')[-2].replace('s', ''))
        id_results_df = pd.read_csv(results_file, sep='\t', index_col=0)
        cancer_type_results_df = calculate_metrics_for_cancer_type(id_results_df,
                                                                   training_data,
                                                                   signal,
                                                                   seed,
                                                                   sample_info_df,
                                                                   classify=classify)
        results_df = pd.concat((results_df, cancer_type_results_df))
    return results_df


def calculate_metrics_for_cancer_type(id_results_df,
                                      training_data,
                                      signal,
                                      seed,
                                      sample_info_df,
                                      classify=True):
    cancer_type_results = []
    for fold in id_results_df.fold_no.unique():
        fold_df = (id_results_df[id_results_df.fold_no == fold]
            .merge(sample_info_df, left_index=True, right_index=True)
            .drop(columns=['sample_type', 'id_for_stratification'])
        )
        for cancer_type in fold_df.cancer_type.unique():
            samples_df = fold_df[fold_df.cancer_type == cancer_type]

            if classify:
                from mpmp.prediction.classification import get_threshold_metrics
                try:
                    with warnings.catch_warnings():
                        # get rid of ROC/PR sample imbalance warnings, we'll catch
                        # that case below
                        warnings.filterwarnings('ignore',
                                                message='No negative samples')
                        warnings.filterwarnings('ignore',
                                                message='No positive samples')
                        warnings.filterwarnings('ignore',
                                                message='invalid value encountered')
                        aupr = (
                            get_threshold_metrics(samples_df.true_class,
                                                  samples_df.positive_prob)
                        )['aupr']
                        auroc = (
                            get_threshold_metrics(samples_df.true_class,
                                                  samples_df.positive_prob)
                        )['auroc']
                except ValueError: # only one class in y_true
                    aupr = np.nan
                    auroc = np.nan
                metric_names = ['aupr', 'auroc']
                cancer_type_results.append((training_data, signal, seed,
                                            fold, cancer_type, aupr, auroc))

            else:
                from mpmp.prediction.regression import get_continuous_metrics
                metrics = get_continuous_metrics(samples_df.true_label,
                                                 samples_df.predicted_output)
                rmse = metrics['rmse']
                r2 = metrics['r2']
                metric_names = ['rmse', 'r2']
                cancer_type_results.append((training_data, signal, seed,
                                            fold, cancer_type, rmse, r2))
    return pd.DataFrame(cancer_type_results,
                        columns=['training_data', 'signal', 'seed',
                                 'fold_no', 'cancer_type'] + metric_names)


def check_compressed_file(results_filename):
    """Check if results file is from compressed experiments."""

    def string_is_int(s):
        # https://stackoverflow.com/a/1267145
        try:
            int(s)
            return True
        except ValueError:
            return False

    # if a file uses compressed data, one component of the filename
    # should have the format 'n{integer}'
    for rs in results_filename.split('_'):
        if rs.startswith('n') and string_is_int(rs.split('n')[1]):
            return True
    return False


def load_preds_to_matrix(preds_dir,
                         sample_info_df,
                         training_data='expression'):
    """Load model predictions into a heatmap/confusion matrix.

    Arguments
    ---------
    preds_dir (str): directory where preds files are located
    sample_info_df (pd.DataFrame): dataframe containing sample information
    training_data (str): type of training data to filter to, if None don't
                         filter

    Returns
    ---------
    preds_df (str): a cancer type x cancer type dataframe, index contains
                    target label and columns are true labels, cells contain
                    average positive class probability for a model trained on
                    the target label and evaluated on the true label (high
                    probability = model predicts column class when trained on
                    row class)
    """
    preds_df = pd.DataFrame()
    for identifier in Path(preds_dir).iterdir():
        identifier_dir = Path(preds_dir, identifier)
        if identifier_dir.is_file():
            continue
        for results_file in identifier_dir.iterdir():
            if not results_file.is_file():
                continue
            results_filename = str(results_file.stem)
            if 'preds' not in results_filename:
                continue
            if 'signal' not in results_filename:
                continue
            if (training_data is not None and
                training_data not in results_filename):
                continue
            cancer_type_preds_df = (
                pd.read_csv(results_file, sep='\t', index_col=0)
                  .merge(sample_info_df[['cancer_type']],
                         left_index=True, right_index=True)
                  .drop(columns=['fold_no', 'true_class'])
                  .groupby('cancer_type')
                  .mean()
                  .T
                  .rename(index={'positive_prob': results_filename.split('_')[0]})
            )
            preds_df = pd.concat((preds_df, cancer_type_preds_df))
    return preds_df.sort_index()


def compare_results(single_cancer_df,
                    pancancer_df=None,
                    identifier='gene',
                    metric='auroc',
                    correction=False,
                    correction_method='fdr_bh',
                    correction_alpha=0.05,
                    verbose=False):
    """Compare cross-validation results between two experimental conditions.

    Main uses for this are comparing an experiment against its negative control
    (shuffled labels), and for comparing two experimental conditions against
    one another.

    Note that this currently uses an unpaired t-test to compare results.
    TODO this could probably use a paired t-test, but need to verify that
    CV folds are actually the same between runs

    Arguments
    ---------
    single_cancer_df (pd.DataFrame): either a single dataframe to compare against
                                     its negative control, or the single-cancer
                                     dataframe
    pancancer_df (pd.DataFrame): if provided, a second dataframe to compare against
                                 single_cancer_df
    identifier (str): column to use as the sample identifier
    metric (str): column to use as the evaluation metric
    correction (bool): whether or not to use a multiple testing correction
    correction_method (str): which method to use for multiple testing correction
                             (from options in statsmodels.stats.multitest)
    correction_alpha (float): significance cutoff to use
    verbose (bool): if True, print verbose output to stderr

    Returns
    -------
    results_df (pd.DataFrame): identifiers and results of statistical test
    """
    if pancancer_df is None:
        results_df = compare_control(single_cancer_df, identifier, metric, verbose)
    else:
        results_df = compare_experiment(single_cancer_df, pancancer_df,
                                        identifier, metric, verbose)
    if correction:
        from statsmodels.stats.multitest import multipletests
        corr = multipletests(results_df['p_value'],
                             alpha=correction_alpha,
                             method=correction_method)
        results_df = results_df.assign(corr_pval=corr[1], reject_null=corr[0])

    return results_df


def compare_control(results_df,
                    identifier='gene',
                    metric='auroc',
                    verbose=False):

    results = []
    unique_identifiers = np.unique(results_df[identifier].values)

    for id_str in unique_identifiers:

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                      (results_df.signal == 'signal'))
        signal_results = results_df[conditions][metric].values
        signal_seeds = results_df[conditions]['seed'].values
        signal_folds = results_df[conditions]['fold'].values

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled'))
        shuffled_results = results_df[conditions][metric].values
        shuffled_seeds = results_df[conditions]['seed'].values
        shuffled_folds = results_df[conditions]['fold'].values

        if signal_results.shape != shuffled_results.shape:
            if verbose:
                print('shapes unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if not (np.array_equal(np.unique(signal_seeds), np.unique(shuffled_seeds))
                and np.array_equal(np.unique(signal_folds), np.unique(shuffled_folds))):
            if verbose:
                print('samples unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if (signal_results.size == 0) or (shuffled_results.size == 0):
            if verbose:
                print('size 0 results array for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        # make sure seeds and folds are in same order
        # this is necessary for paired t-test
        try:
            assert np.array_equal(signal_seeds, shuffled_seeds)
            assert np.array_equal(signal_folds, shuffled_folds)
        except AssertionError:
            print(id_str)
            print(signal_seeds, shuffled_seeds)
            print(signal_folds, shuffled_folds)

        if np.array_equal(signal_results, shuffled_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(signal_results) - np.mean(shuffled_results)
            p_value = ttest_rel(signal_results, shuffled_results)[1]
        results.append([id_str, delta_mean, p_value])

    return pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])


def compare_control_ind(results_df,
                        identifier='gene',
                        metric='auroc',
                        verbose=False):
    """Compare signal vs. shuffled results for each seed/CV fold independently.

    This allows customized statistical analysis after performing comparison
    (as opposed to compare_control which automatically aggregates over
    seeds/folds).
    """

    results = []
    unique_identifiers = np.unique(results_df[identifier].values)

    for id_str in unique_identifiers:

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                      (results_df.signal == 'signal'))
        signal_results = results_df[conditions].copy()

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled'))
        shuffled_results = results_df[conditions].copy()

        if signal_results.shape != shuffled_results.shape:
            if verbose:
                print('shapes unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if (signal_results.size == 0) or (shuffled_results.size == 0):
            if verbose:
                print('size 0 results array for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        for seed in results_df.seed.unique():
            for fold in results_df.fold.unique():
                try:
                    signal_value = signal_results[(signal_results.seed == seed) &
                                                 (signal_results.fold == fold)][metric].values[0]
                    shuffled_value = shuffled_results[(shuffled_results.seed == seed) &
                                                      (shuffled_results.fold == fold)][metric].values[0]
                    delta = signal_value - shuffled_value
                    results.append([id_str, seed, fold, delta])
                except IndexError:
                    # this seed/fold combo doesn't exist, just skip it
                    continue

    return pd.DataFrame(results,
                        columns=['identifier', 'seed', 'fold',
                                'delta_{}'.format(metric)])


def compare_experiment(single_cancer_df,
                       pancancer_df,
                       identifier='gene',
                       metric='auroc',
                       verbose=False):

    results = []
    single_cancer_ids = np.unique(single_cancer_df[identifier].values)
    pancancer_ids = np.unique(pancancer_df[identifier].values)
    unique_identifiers = list(set(single_cancer_ids).intersection(pancancer_ids))

    for id_str in unique_identifiers:

        conditions = ((single_cancer_df[identifier] == id_str) &
                      (single_cancer_df.data_type == 'test') &
                      (single_cancer_df.signal == 'signal'))
        single_cancer_results = single_cancer_df[conditions][metric].values

        conditions = ((pancancer_df[identifier] == id_str) &
                      (pancancer_df.data_type == 'test') &
                      (pancancer_df.signal == 'signal'))
        pancancer_results = pancancer_df[conditions][metric].values

        if single_cancer_results.shape != pancancer_results.shape:
            if verbose:
                print('shapes unequal for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        if (single_cancer_results.size == 0) or (pancancer_results.size == 0):
            if verbose:
                print('size 0 results array for {}, skipping'.format(id_str),
                      file=sys.stderr)
            continue

        delta_mean = np.mean(pancancer_results) - np.mean(single_cancer_results)
        if np.array_equal(pancancer_results, single_cancer_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(pancancer_results) - np.mean(single_cancer_results)
            p_value = ttest_rel(pancancer_results, single_cancer_results)[1]
        results.append([id_str, delta_mean, p_value])

    return pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])


def generate_nonzero_coefficients(results_dir):
    """Generate coefficients from mutation prediction model fits.

    Loading all coefficients into memory at once is prohibitive, so we generate
    them individually and analyze/summarize in analysis scripts.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes

    Yields
    ------
    identifier (str): identifier for given coefficients
    coefs (dict): list of nonzero coefficients for each fold of CV, for the
                  given identifier
    """
    coefs = {}
    all_features = None
    for gene_name in os.listdir(results_dir):
        gene_dir = os.path.join(results_dir, gene_name)
        if not os.path.isdir(gene_dir): continue
        for coefs_file in os.listdir(gene_dir):
            if coefs_file[0] == '.': continue
            if 'signal' not in coefs_file: continue
            if 'coefficients' not in coefs_file: continue
            training_data = coefs_file.split('_')[1]
            full_coefs_file = os.path.join(gene_dir, coefs_file)
            coefs_df = pd.read_csv(full_coefs_file, sep='\t')
            if all_features is None:
                all_features = np.unique(coefs_df.feature.values)
            identifier = '{}_{}'.format(gene_name, training_data)
            coefs = process_coefs(coefs_df)
            yield identifier, coefs


def process_coefs(coefs_df):
    """Process and return nonzero coefficients for a single identifier"""
    id_coefs = []
    for fold in np.sort(np.unique(coefs_df.fold.values)):
        conditions = ((coefs_df.fold == fold) &
                      (coefs_df['abs'] > 0))
        nz_coefs_df = coefs_df[conditions]
        id_coefs.append(list(zip(nz_coefs_df.feature.values,
                                 nz_coefs_df.weight.values)))
    return id_coefs


