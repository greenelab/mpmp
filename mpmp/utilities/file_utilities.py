"""
Functions for writing and processing output files
"""
from pathlib import Path

import pandas as pd

from mpmp.exceptions import ResultsFileExistsError

def make_output_dir(results_dir, identifier, exp_string='cancer_type'):
    """Create a directory to write output to."""
    output_dir = Path(results_dir, exp_string, identifier).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def check_output_file(output_dir,
                      identifier,
                      training_data,
                      shuffle_labels,
                      seed):
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = Path(output_dir,
                      "{}_{}_{}_s{}_coefficients.tsv.gz".format(
                          identifier, training_data, signal, seed)).resolve()
    if check_file.is_file():
        raise ResultsFileExistsError(
            'Results file already exists for identifier: {}\n'.format(
                identifier)
        )
    return check_file


def save_results(output_dir,
                 check_file,
                 results,
                 exp_string,
                 identifier,
                 training_data,
                 shuffle_labels,
                 seed):

    signal = 'shuffled' if shuffle_labels else 'signal'
    auc_df = pd.concat(results[
        '{}_auc'.format(exp_string)
    ])
    aupr_df = pd.concat(results[
        '{}_aupr'.format(exp_string)
    ])
    coef_df = pd.concat(results[
        '{}_coef'.format(exp_string)
    ])
    metrics_df = pd.concat(results[
        '{}_metrics'.format(exp_string)
    ])

    if '{}_preds'.format(exp_string) in results:
        preds_df = pd.concat(results[
            '{}_preds'.format(exp_string)
        ])
    else:
        preds_df = None

    coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = Path(
        output_dir, "{}_{}_{}_s{}_auc_threshold_metrics.tsv.gz".format(
            identifier, training_data, signal, seed)).resolve()
    auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        output_dir, "{}_{}_{}_s{}_aupr_threshold_metrics.tsv.gz".format(
            identifier, training_data, signal, seed)).resolve()
    aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        output_dir, "{}_{}_{}_s{}_classify_metrics.tsv.gz".format(
            identifier, training_data, signal, seed)).resolve()
    metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    if preds_df is not None:
        output_file = Path(
            output_dir, "{}_{}_{}_s{}_preds.tsv.gz".format(
                identifier, training_data, signal, seed)).resolve()
        preds_df.to_csv(
            output_file, sep="\t", compression="gzip", float_format="%.5g"
        )


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)

