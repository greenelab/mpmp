"""
Functions for writing and processing output files
"""
from pathlib import Path

import pandas as pd

from mpmp.exceptions import ResultsFileExistsError

def make_cancer_type_dir(results_dir, cancer_type):
    """Create a directory for the given cancer type."""
    dirname = 'cancer_type'
    cancer_type_dir = Path(results_dir, dirname, cancer_type).resolve()
    cancer_type_dir.mkdir(parents=True, exist_ok=True)
    return cancer_type_dir


def check_cancer_type_file(cancer_type_dir,
                           cancer_type,
                           training_data,
                           shuffle_labels,
                           seed):

    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = Path(cancer_type_dir,
                      "{}_{}_{}_s{}_coefficients.tsv.gz".format(
                          cancer_type, training_data, signal, seed)).resolve()
    if check_file.is_file():
        raise ResultsFileExistsError(
            'Results file already exists for cancer type: {}\n'.format(
                cancer_type)
        )
    return check_file


def save_results_cancer_type(cancer_type_dir,
                             check_file,
                             results,
                             cancer_type,
                             training_data,
                             shuffle_labels,
                             seed):

    signal = 'shuffled' if shuffle_labels else 'signal'
    cancer_type_auc_df = pd.concat(results['cancer_type_auc'])
    cancer_type_aupr_df = pd.concat(results['cancer_type_aupr'])
    cancer_type_coef_df = pd.concat(results['cancer_type_coef'])
    cancer_type_metrics_df = pd.concat(results['cancer_type_metrics'])

    cancer_type_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip",
        float_format="%.5g"
    )

    output_file = Path(
        cancer_type_dir, "{}_{}_{}_s{}_auc_threshold_metrics.tsv.gz".format(
            cancer_type, training_data, signal, seed)).resolve()
    cancer_type_auc_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        cancer_type_dir, "{}_{}_{}_s{}_aupr_threshold_metrics.tsv.gz".format(
            cancer_type, training_data, signal, seed)).resolve()
    cancer_type_aupr_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    output_file = Path(
        cancer_type_dir, "{}_{}_{}_s{}_classify_metrics.tsv.gz".format(
            cancer_type, training_data, signal, seed)).resolve()
    cancer_type_metrics_df.to_csv(
        output_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)

