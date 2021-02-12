import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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
            if 'classify' not in results_filename: continue
            if results_filename[0] == '.': continue
            id_results_df = pd.read_csv(results_file, sep='\t')
            id_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, id_results_df))
    return results_df


def load_compressed_prediction_results(results_dir, experiment_descriptor):
    """Load results of compressed prediction experiments.

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
            if 'classify' not in results_filename: continue
            if results_filename[0] == '.': continue
            n_dims = int(results_filename.split('_')[-3].replace('n', ''))
            id_results_df = pd.read_csv(results_file, sep='\t')
            id_results_df['n_dims'] = n_dims
            id_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, id_results_df))
    return results_df


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

        conditions = ((results_df[identifier] == id_str) &
                      (results_df.data_type == 'test') &
                     (results_df.signal == 'shuffled'))
        shuffled_results = results_df[conditions][metric].values

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

        if np.array_equal(signal_results, shuffled_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(signal_results) - np.mean(shuffled_results)
            p_value = ttest_ind(signal_results, shuffled_results)[1]
        results.append([id_str, delta_mean, p_value])

    return pd.DataFrame(results, columns=['identifier', 'delta_mean', 'p_value'])


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
            p_value = ttest_ind(pancancer_results, single_cancer_results)[1]
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


