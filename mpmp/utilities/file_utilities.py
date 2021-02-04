"""
Functions for writing and processing output files
"""
from pathlib import Path
import pickle as pkl

import pandas as pd

from mpmp.exceptions import ResultsFileExistsError

def make_output_dir(experiment_dir, identifier):
    """Create a directory to write output to."""
    output_dir = Path(experiment_dir, identifier).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def construct_filename(output_dir,
                       identifier,
                       shuffle_labels,
                       model_options,
                       file_descriptor,
                       extension):
    """Construct a filename from experimental parameters.

    This allows us to standardize what information goes into the filename
    (i.e. what information constitutes a unique "experiment").
    """
    training_data = model_options.training_data
    signal = 'shuffled' if shuffle_labels else 'signal'
    seed = model_options.seed
    if identifier is None:
        # this might be the case for files that pertain to the whole experiment
        # (e.g. metadata/parameters file)
        try:
            return Path(output_dir,
                        '{}_s{}_n{}_{}{}'.format(training_data,
                                                 seed,
                                                 model_options.n_dim,
                                                 file_descriptor,
                                                 extension))
        except AttributeError:
            # no n_dim in model_options => not a compressed model
            return Path(output_dir,
                        '{}_s{}_{}{}'.format(training_data,
                                             seed,
                                             file_descriptor,
                                             extension))
    else:
        # TODO probably a better way to do this
        try:
            return Path(output_dir,
                        '{}_{}_{}_s{}_n{}_{}{}'.format(identifier,
                                                       training_data,
                                                       signal,
                                                       seed,
                                                       model_options.n_dim,
                                                       file_descriptor,
                                                       extension))
        except AttributeError:
            # no n_dim in model_options => not a compressed model
            return Path(output_dir,
                        '{}_{}_{}_s{}_{}{}'.format(identifier,
                                                   training_data,
                                                   signal,
                                                   seed,
                                                   file_descriptor,
                                                   extension))


def save_model_options(output_dir, model_options):
    """Save model hyperparameters/metadata to output directory.

    model_options is an argparse Namespace, and is converted to a dictionary
    and pickled.
    """
    output_file = construct_filename(output_dir,
                                     None,
                                     None,
                                     model_options,
                                     'model_options',
                                     '.pkl')
    with open(output_file, 'wb') as f:
        pkl.dump(vars(model_options), f)


def check_output_file(output_dir,
                      identifier,
                      shuffle_labels,
                      model_options):
    """Check if results already exist for a given experiment identifier.

    If the file does not exist, return the filename.
    """

    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = construct_filename(output_dir,
                                    identifier,
                                    shuffle_labels,
                                    model_options,
                                    'coefficients',
                                    '.tsv.gz')
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
                 shuffle_labels,
                 model_options):
    """Save results of a single experiment for a single identifier."""

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
        check_file, sep="\t", index=False, float_format="%.5g"
    )

    output_file = construct_filename(output_dir,
                                     identifier,
                                     shuffle_labels,
                                     model_options,
                                     'auc_threshold_metrics',
                                     '.tsv.gz')
    auc_df.to_csv(
        output_file, sep="\t", index=False, float_format="%.5g"
    )

    output_file = construct_filename(output_dir,
                                     identifier,
                                     shuffle_labels,
                                     model_options,
                                     'aupr_threshold_metrics',
                                     '.tsv.gz')
    aupr_df.to_csv(
        output_file, sep="\t", index=False, float_format="%.5g"
    )

    output_file = construct_filename(output_dir,
                                     identifier,
                                     shuffle_labels,
                                     model_options,
                                     'classify_metrics',
                                     '.tsv.gz')
    metrics_df.to_csv(
        output_file, sep="\t", index=False, float_format="%.5g"
    )

    if preds_df is not None:
        output_file = construct_filename(output_dir,
                                         identifier,
                                         shuffle_labels,
                                         model_options,
                                         'preds',
                                         '.tsv.gz')
        preds_df.to_csv(
            output_file, sep="\t", float_format="%.5g"
        )


def generate_log_df(log_columns, log_values):
    """Generate and format log output."""
    return pd.DataFrame(dict(zip(log_columns, log_values)), index=[0])


def write_log_file(log_df, log_file):
    """Append log output to log file."""
    log_df.to_csv(log_file, mode='a', sep='\t', index=False, header=False)

