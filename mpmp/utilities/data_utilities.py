"""
Functions for reading and processing input data

"""
import os
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import mpmp.config as cfg
from mpmp.utilities.tcga_utilities import (
    compress_and_save_data,
    get_compress_output_prefix
)

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


def load_compressed_data(data_type,
                         n_dim,
                         verbose=False,
                         standardize_input=False,
                         load_subset=False):
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
    if load_subset:
        raise NotImplementedError('no subsampled compressed data')
    try:
        output_prefix = get_compress_output_prefix(data_type,
                                                   n_dim,
                                                   cfg.default_seed,
                                                   standardize_input)
        data_df = pd.read_csv(
            os.path.join(cfg.compressed_data_dir, '{}.tsv.gz'.format(output_prefix)),
            index_col=0, sep='\t'
        )
    except OSError:
        # file doesn't exist so we have to create it
        if verbose:
            print('PCA compressing data type: {}, n_dims: {}'.format(
                data_type, n_dim))
        data_df = compress_and_save_data(
            data_type,
            load_raw_data(data_type),
            cfg.compressed_data_dir,
            n_dim,
            standardize_input,
            verbose
        )
    return data_df


def load_multiple_data_types(data_types, n_dims, standardize_input, verbose=False):
    """Load multiple data types and concatenate columns.

    Arguments
    ---------
    data_types (list): list of data types to use
    n_dims (list): list of compressed dimensions to use, None for raw data
    verbose (bool): whether or not to print verbose output

    Returns
    -------
    data_df: samples x latent dimensions dataframe
    data_ixs: 1D numpy array containing data type index for each column
    """
    data_df = pd.DataFrame()
    data_ixs = []
    if verbose:
        print('Loading data for data types {}...'.format(', '.join(data_types)),
              file=sys.stderr)
    if standardize_input is None:
        standardize_input = [False] * len(data_types)
    for ix, (data_type, n_dim, standardize) in enumerate(zip(data_types, n_dims, standardize_input)):
        if verbose:
            print('- Loading {} data...'.format(data_type), file=sys.stderr)
        if n_dim is not None:
            partial_data_df = load_compressed_data(data_type, int(n_dim),
                                                   standardize_input=standardize)
            # the default PC name is just the integer index, so here we need
            # to rename them in case we have PCs for more than one data type
            #
            # just rename them to {data_type}_pc{pc_number}
            partial_data_df = partial_data_df.add_prefix(
                                  '{}_pc'.format(data_type))
        else:
            partial_data_df = load_raw_data(data_type)
        data_ixs += [ix] * partial_data_df.shape[1]
        # only keep samples that are in all data types (inner join)
        data_df = pd.concat((data_df, partial_data_df), axis=1,
                            join=('outer' if ix == 0 else 'inner'))
    return data_df, np.array(data_ixs)


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


def load_top_genes():
    """Load top mutated genes in TCGA.

    These are precomputed in 00_download_data/sample_random_genes.ipynb.
    """
    genes_df = pd.read_csv(cfg.top_genes, sep='\t')
    return genes_df


def load_random_genes():
    """Load randomly sampled genes.

    These are sampled in 00_download_data/sample_random_genes.ipynb; criteria
    for sampling are described in that notebook.
    """
    return pd.read_csv(cfg.random_genes, sep='\t')


def load_vogelstein():
    """Load list of cancer-relevant genes from Vogelstein et al. 2013
    (https://doi.org/10.1126/science.1235122).

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
    # some genes in vogelstein set have different names in mutation data
    genes_df.gene.replace(to_replace=cfg.gene_aliases, inplace=True)
    return genes_df


def load_cosmic():
    """Load list of cancer-relevant genes from COSMIC Cancer Gene Census
    (https://cancer.sanger.ac.uk/cosmic/census?tier=1).

    These genes and their oncogene or TSG status are precomputed in
    the explore_cosmic_gene_set notebook.
    """

    genes_df = pd.read_csv(cfg.cosmic_with_annotations, sep='\t')

    # some genes in cosmic set have different names in mutation data
    genes_df.gene.replace(to_replace=cfg.gene_aliases, inplace=True)
    return genes_df


def load_custom_genes(gene_set):
    """Load oncogene/TSG annotation information for custom genes.

    This will load annotations from the gene sets corresponding to
    load_functions, in that order of priority.
    """
    # make sure the passed-in gene set is a list
    assert isinstance(gene_set, typing.List)

    load_functions = [
        load_vogelstein,
        load_top_genes,
        load_random_genes,
        load_cosmic
    ]
    genes_df = None
    for load_fn in load_functions:
        annotated_df = load_fn()
        if set(gene_set).issubset(set(annotated_df.gene.values)):
            genes_df = annotated_df[annotated_df.gene.isin(gene_set)]
            break

    if genes_df is None:
        # note that this will happen if gene_set is not a subset of exactly
        # one of the gene sets in load_functions
        #
        # we could allow gene_set to be a subset of the union of all of them,
        # but that would take a bit more work and is probably not a super
        # common use case for us
        from mpmp.exceptions import GenesNotFoundError
        raise GenesNotFoundError(
            'Gene list was not a subset of any existing gene set'
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
            genes_df = load_top_genes()
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
    try:
        return pd.read_csv(cfg.sample_infos[train_data_type],
                           sep='\t', index_col='sample_id')
    except KeyError as e:
        if 'mutations' in train_data_type:
            return pd.read_csv(cfg.sample_infos['mutation'],
                               sep='\t', index_col='sample_id')
        elif 'mutation_preds' in train_data_type:
            # if we're using predicted mutation scores, just get the sample
            # info from the source data type
            return pd.read_csv(
                cfg.sample_infos[train_data_type.replace('mutation_preds_', '')],
                sep='\t', index_col='sample_id'
            )
        else:
            raise e


def load_sample_info_multi(train_data_types, verbose=False):
    if verbose:
        print('Loading sample info for multiple data types...',
              file=sys.stderr)
    sample_info_df = load_sample_info(train_data_types[0])
    for training_data in train_data_types[1:]:
        # add the rows that are not in current sample info
        add_df = load_sample_info(training_data)
        add_df = add_df[~add_df.index.isin(sample_info_df.index)]
        sample_info_df = pd.concat((sample_info_df, add_df))
    return sample_info_df


def load_significant_genes(sample_set='all'):
    if sample_set == 'methylation':
        significance_df = pd.read_csv(cfg.sig_genes_methylation, sep='\t')
    else:
        significance_df = pd.read_csv(cfg.sig_genes_all, sep='\t')
    return significance_df.loc[significance_df.reject_null, 'gene'].values


def load_mutation_predictions(train_data_type):
    source_data = train_data_type.replace('mutation_preds_', '')
    return pd.read_csv(cfg.predictions[source_data],
                       index_col=0, sep='\t')


def load_purity(mut_burden_df,
                sample_info_df,
                classify=False,
                verbose=False):
    """Load tumor purity data.

    Arguments
    ---------
    mut_burden_df (pd.DataFrame): dataframe with sample mutation burden info
    sample_info_df (pd.DataFrame): dataframe with sample cancer type info
    classify (bool): if True, binarize tumor purity values above/below median
    verbose (bool): if True, print verbose output

    Returns
    -------
    purity_df (pd.DataFrame): dataframe where the "status" attribute is purity
    """

    if verbose:
        print('Loading tumor purity info...', file=sys.stderr)

    # some samples don't have purity calls, we can just drop them
    purity_df = (
        pd.read_csv(cfg.tumor_purity_data,sep='\t', index_col='array')
          .dropna(subset=['purity'])
    )
    purity_df.index.rename('sample_id', inplace=True)

    # for classification, we want to binarize purity values into above/below
    # the median (1 = above, 0 = below; this is arbitrary)
    if classify:
        purity_df['purity'] = (
            purity_df.purity > purity_df.purity.median()
        ).astype('int')

    # join mutation burden information and cancer type information
    # these are necessary to generate non-gene covariates later on
    purity_df = (purity_df
        .merge(mut_burden_df, left_index=True, right_index=True)
        .merge(sample_info_df, left_index=True, right_index=True)
        .rename(columns={'cancer_type': 'DISEASE',
                         'purity': 'status'})
    )
    return purity_df.loc[:, ['status', 'DISEASE', 'log10_mut']]


def load_msi(cancer_type, mut_burden_df, sample_info_df, verbose=False):
    """Load microsatellite instability data.

    Arguments
    ---------
    mut_burden_df (pd.DataFrame): dataframe with sample mutation burden info
    sample_info_df (pd.DataFrame): dataframe with sample cancer type info
    verbose (bool): if True, print verbose output

    Returns
    -------
    msi_df (pd.DataFrame): dataframe where the "status" attribute is a binary
                           label (1 = MSI-H, 0 = anything else)
    """

    if verbose:
        print('Loading microsatellite instability info...', file=sys.stderr)

    if cancer_type == 'pancancer':
        msi_df = _load_msi_all()
    else:
        msi_df = _load_msi_cancer_type(cancer_type)

    msi_df.index.rename('sample_id', inplace=True)

    # do one-vs-rest classification, with the MSI-high subtype as positive
    # label and everything alse (MSI-low, MSS, undetermined) as negatives
    msi_df['status'] = (msi_df.msi_status == 'msi-h').values.astype('int')


    # clinical data is identified by the patient info (without the sample
    # ID), so we want to match the first ten characters in the other dataframes
    mut_burden_df['sample_first_ten'] = (
        mut_burden_df.index.to_series().str.split('-').str[:3].str.join('-')
    )

    # join mutation burden information and MSI information
    # these are necessary to generate non-gene covariates later on
    msi_df = (msi_df
        .drop(columns=['msi_status'])
        .merge(mut_burden_df, left_index=True, right_on='sample_first_ten')
        .merge(sample_info_df, left_index=True, right_index=True)
        .rename(columns={'cancer_type': 'DISEASE'})
    )
    return msi_df.loc[:, ['status', 'DISEASE', 'log10_mut']]


def _load_msi_all():
    msi_list = []
    for cancer_type in cfg.msi_cancer_types:
        msi_list.append(_load_msi_cancer_type(cancer_type))
    return pd.concat(msi_list)


def _load_msi_cancer_type(cancer_type):
    return pd.read_csv(Path(cfg.msi_data_dir,
                            '{}_msi_status.tsv'.format(cancer_type)),
                       sep='\t', index_col=0)


def load_survival_labels(cancer_type,
                         mut_burden_df,
                         sample_info_df,
                         verbose=False):
    """Load data relevant to survival prediction.

    Arguments
    ---------
    mut_burden_df (pd.DataFrame): dataframe with sample mutation burden info
    sample_info_df (pd.DataFrame): dataframe with sample cancer type info
    verbose (bool): if True, print verbose output

    Returns
    -------
    survival_df (pd.DataFrame): dataframe where the "status" attribute is
                                the relevant survival duration
    """

    if verbose:
        print('Loading survival info and covariates...', file=sys.stderr)

    clinical_df = (
        pd.read_excel(cfg.clinical_data,
                      sheet_name='TCGA-CDR',
                      index_col='bcr_patient_barcode',
                      engine='openpyxl')
    )
    clinical_df.index.rename('sample_id', inplace=True)

    # drop numeric index column
    clinical_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

    # we want to use age as a covariate
    clinical_df.rename(columns={'age_at_initial_pathologic_diagnosis': 'age'},
                       inplace=True)

    # join mutation burden information and cancer type information
    # these are necessary to generate non-gene covariates later on
    covariate_df = (mut_burden_df
        .merge(sample_info_df, left_index=True, right_index=True)
        .rename(columns={'cancer_type': 'DISEASE'})
    )
    # clinical data omits the tumor information ('-01', etc) at the end
    # of the sample identifier
    covariate_df['clinical_id'] = covariate_df.index.str[:-3]
    clinical_df = (clinical_df
        .merge(covariate_df, left_index=True, right_on='clinical_id')
    )

    assert clinical_df.clinical_id.duplicated().sum() == 0

    # we want to use overall survival as the target variable except for
    # certain cancer types where progression-free intervals are typically
    # used (since very few deaths are observed)
    # this is recommended in https://doi.org/10.1016/j.cell.2018.02.052
    clinical_df['time_in_days'] = clinical_df['OS.time']
    clinical_df['status'] = clinical_df['OS'].astype('bool')
    pfi_samples = clinical_df.DISEASE.isin(cfg.pfi_cancer_types)
    clinical_df.loc[pfi_samples, 'time_in_days'] = clinical_df[pfi_samples]['PFI.time']
    clinical_df.loc[pfi_samples, 'status'] = clinical_df[pfi_samples]['PFI'].astype('bool')

    # clean up columns and drop samples with NA survival times
    na_survival_times = (clinical_df['time_in_days'].isna())
    cols_to_keep = ['status', 'time_in_days', 'age', 'DISEASE', 'log10_mut']
    clinical_df = clinical_df.loc[~na_survival_times, cols_to_keep].copy()

    # mean impute missing age values for now
    # TODO: we could do this by cancer type
    clinical_df.age.fillna(clinical_df.age.mean(), inplace=True)

    if cancer_type == 'pancancer':
        return clinical_df
    else:
        cancer_type_samples = (clinical_df.DISEASE == cancer_type)
        return clinical_df.loc[cancer_type_samples, :].copy()


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

