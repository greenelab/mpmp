"""
Functions for reading and processing input data

"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import mpmp.config as cfg

def load_raw_data(train_data_type, verbose=False, load_subset=False):
    """Load and preprocess saved TCGA data.

    Arguments
    ---------
    train_data_type (str): type of data to load, options in config
    verbose (bool): whether or not to print verbose output
    load_subset (bool): whether or not to subset data for faster debugging

    Returns
    -------
    data_df: samples x features dataframe
    """
    if load_subset:
        if verbose:
            print(
                'Loading subset of {} data for debugging...'.format(train_data_type),
                file=sys.stderr
            )
        try:
            data_df = pd.read_csv(cfg.subsampled_data_types[train_data_type],
                                  index_col=0, sep='\t')
        except KeyError:
            raise NotImplementedError('No debugging subset generated for '
                                      '{} data'.format(train_data_type))
    else:
        if verbose:
            print(
                'Loading {} data...'.format(train_data_type),
                file=sys.stderr
            )
        if train_data_type == 'me_450k':
            # since this dataset is so large, storing it as a pickle is
            # much faster than loading it from a .tsv or .gz file
            data_df = pd.read_pickle(cfg.data_types[train_data_type])
        else:
            data_df = pd.read_csv(cfg.data_types[train_data_type],
                                  index_col=0, sep='\t')
    return data_df


def load_compressed_data(data_type, n_dim, verbose=False, debug=False):
    """Load compressed data for the given data type and compressed dimensions.

    Arguments
    ---------
    data_type (str): data type to use
    n_dim (int): number of latent dimensions to use
    verbose (bool): whether or not to print verbose output
    debug (bool): whether or not to subset data for faster debugging

    Returns
    -------
    data_df: samples x latent dimensions dataframe
    """
    if debug:
        raise NotImplementedError('no subsampled compressed data')
    try:
        data_df = pd.read_csv(
            str(cfg.compressed_data_types[data_type]).format(n_dim),
            index_col=0, sep='\t'
        )
    except OSError:
        # compressed data does not exist for given n_dim
        raise NotImplementedError(
            'no compressed data for data_type {}, n_dim {}'.format(
                data_type, n_dim)
        )
    return data_df


def load_pancancer_data(verbose=False, test=False, subset_columns=None):
    """Load pan-cancer relevant data from previous Greene Lab repos.

    Data being loaded includes:
    * sample_freeze_df: list of samples from TCGA "data freeze" in 2017
    * mutation_df: deleterious mutation count information for freeze samples
      (this is a samples x genes dataframe, entries are the number of
       deleterious mutations in the given gene for the given sample)
    * copy_loss_df: copy number loss information for freeze samples
    * copy_gain_df: copy number gain information for freeze samples
    * mut_burden_df: log10(total deleterious mutations) for freeze samples

    Most of this data was originally compiled and documented in Greg's
    pancancer repo: http://github.com/greenelab/pancancer
    See, e.g.
    https://github.com/greenelab/pancancer/blob/master/scripts/initialize/process_sample_freeze.py
    for more info on mutation processing steps.

    Arguments
    ---------
    verbose (bool): whether or not to print verbose output
    test (bool): whether or not to load data subset for testing

    Returns
    -------
    pancan_data: TCGA "data freeze" mutation information described above
    """

    # loading this data from the pancancer repo is very slow, so we
    # cache it in a pickle to speed up loading
    if test:
        import mpmp.test_config as tcfg
        data_filepath = tcfg.test_pancan_data
    else:
        data_filepath = cfg.pancan_data

    if os.path.exists(data_filepath):
        if verbose:
            print('Loading pan-cancer data from cached pickle file...', file=sys.stderr)
        with open(data_filepath, 'rb') as f:
            pancan_data = pkl.load(f)
    else:
        if verbose:
            print('Loading pan-cancer data from repo (warning: slow)...', file=sys.stderr)
        pancan_data = load_pancancer_data_from_repo(subset_columns)
        with open(data_filepath, 'wb') as f:
            pkl.dump(pancan_data, f)

    return pancan_data


def load_top_50():
    """Load top 50 mutated genes in TCGA from BioBombe repo.

    These were precomputed for the equivalent experiments in the
    BioBombe paper, so no need to recompute them.
    """
    file = "{}/{}/9.tcga-classify/data/top50_mutated_genes.tsv".format(
            cfg.top50_base_url, cfg.top50_commit)
    genes_df = pd.read_csv(file, sep='\t')
    return genes_df


def load_vogelstein():
    """Load list of cancer-relevant genes from Vogelstein and Kinzler,
    Nature Medicine 2004 (https://doi.org/10.1038/nm1087)

    These genes and their oncogene or TSG status were precomputed in
    the pancancer repo, so we just load them from there.
    """

    file = "{}/{}/data/vogelstein_cancergenes.tsv".format(
            cfg.vogelstein_base_url, cfg.vogelstein_commit)
    genes_df = (
        pd.read_csv(file, sep='\t')
          .rename(columns={'Gene Symbol'   : 'gene',
                           'Classification*': 'classification'})
    )
    return genes_df


def get_classification(gene, genes_df=None):
    """Get oncogene/TSG classification from existing datasets for given gene."""
    classification = 'neither'
    if (genes_df is not None) and (gene in genes_df.gene):
        classification = genes_df[genes_df.gene == gene].classification.iloc[0]
    else:
        genes_df = load_vogelstein()
        if gene in genes_df.gene:
            classification = genes_df[genes_df.gene == gene].classification.iloc[0]
        else:
            genes_df = load_top_50()
            if gene in genes_df.gene:
                classification = genes_df[genes_df.gene == gene].classification.iloc[0]
    return classification


def load_pancancer_data_from_repo(subset_columns=None):
    """Load data to build feature matrices from pancancer repo. """

    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/sample_freeze.tsv".format(base_url, commit)
    sample_freeze_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/pancan_mutation_freeze.tsv.gz".format(base_url, commit)
    mutation_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_loss_status.tsv.gz".format(base_url, commit)
    copy_loss_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_gain_status.tsv.gz".format(base_url, commit)
    copy_gain_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/mutation_burden_freeze.tsv".format(base_url, commit)
    mut_burden_df = pd.read_csv(file, index_col=0, sep='\t')

    if subset_columns is not None:
        # don't reindex sample_freeze_df or mut_burden_df
        # they don't have gene-valued columns
        mutation_df = mutation_df.reindex(subset_columns, axis='columns')
        copy_loss_df = copy_loss_df.reindex(subset_columns, axis='columns')
        copy_gain_df = copy_gain_df.reindex(subset_columns, axis='columns')

    return (
        sample_freeze_df,
        mutation_df,
        copy_loss_df,
        copy_gain_df,
        mut_burden_df
    )


def load_sample_info(train_data_type, verbose=False):
    if verbose:
        print('Loading sample info...', file=sys.stderr)
    return pd.read_csv(cfg.sample_infos[train_data_type],
                       sep='\t', index_col='sample_id')


def split_argument_groups(args, parser):
    """Split argparse script arguments into argument groups.

    See: https://stackoverflow.com/a/46929320
    """
    import argparse
    arg_groups = {}
    for group in parser._action_groups:
        if group.title in ['positional arguments', 'optional arguments']:
            continue
        group_dict = {
            a.dest : getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return arg_groups

