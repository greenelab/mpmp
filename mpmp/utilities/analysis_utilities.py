import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def load_cancer_type_prediction_results(results_dir, experiment_descriptor):
    """Load results of cancer type prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual cancer types
    experiment_descriptor (str): string describing this experiment, can be
                                 useful to segment analyses involving multiple
                                 experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for cancer_type in results_dir.iterdir():
        cancer_type_dir = Path(results_dir, cancer_type)
        if cancer_type_dir.is_file(): continue
        for results_file in cancer_type_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.stem)
            if 'classify' not in results_filename: continue
            if results_filename[0] == '.': continue
            gene_results_df = pd.read_csv(results_file, sep='\t')
            gene_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, gene_results_df))
    return results_df


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

