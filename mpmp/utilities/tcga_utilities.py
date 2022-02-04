"""
Functions for preprocessing TCGA expression data and mutation status labels.

Most functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import mpmp.config as cfg

def process_y_matrix(y_mutation,
                     y_copy,
                     include_copy,
                     gene,
                     sample_freeze,
                     mutation_burden,
                     filter_count,
                     filter_prop,
                     output_directory,
                     filter_cancer_types=True,
                     hyper_filter=5,
                     test=False,
                     overlap_data_types=None):
    """
    Combine copy number and mutation data and filter cancer-types to build y matrix

    Arguments
    ---------
    y_mutation: Pandas DataFrame of mutation status
    y_copy: Pandas DataFrame of copy number status
    include_copy: boolean if the copy number data should be included in status calc
    gene: string indicating gene of interest (used for writing proportion file)
    sample_freeze: pandas dataframe storing which samples to use
    mutation_burden: pandas dataframe storing log10 mutation counts
    filter_count: the number of positives or negatives required per cancer-type
    filter_prop: the proportion of positives or negatives required per cancer-type
    output_directory: the name of the directory to store the gene summary
    filter_cancer_types: if False, don't filter cancer types that don't meet criteria
    hyper_filter: the number of std dev above log10 mutation burden to filter
    test: if True, don't write filtering info to disk
    overlap_data_types: if not None, use samples present for all included data types

    Returns
    -------
    Write file of cancer-type filtering to disk and output a processed y vector
    """
    if include_copy:
        y_df = y_copy + y_mutation
    else:
        y_df = y_mutation

    y_df.loc[y_df > 1] = 1
    y_df = pd.DataFrame(y_df)
    y_df.columns = ["status"]

    y_df = (
        y_df.merge(
            sample_freeze, how="left", left_index=True, right_on="SAMPLE_BARCODE"
        )
        .set_index("SAMPLE_BARCODE")
        .merge(mutation_burden, left_index=True, right_index=True)
    )

    if overlap_data_types is not None:
        valid_samples = (
            set(get_cross_data_samples(data_types=overlap_data_types))
              .intersection(set(y_df.index))
        )
        # convert back to list, to make the sample order deterministic
        valid_samples = sorted(list(valid_samples))
        y_df = y_df.reindex(valid_samples)
    else:
        valid_samples = None

    # Filter to remove hyper-mutated samples
    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :]

    # Get statistics per gene and disease
    disease_counts_df = pd.DataFrame(y_df.groupby("DISEASE").sum()["status"])

    disease_proportion_df = disease_counts_df.divide(
        y_df["DISEASE"].value_counts(sort=False).sort_index(), axis=0
    )

    # Filter diseases with low counts or proportions for classification balance
    filter_disease_df = (disease_counts_df > filter_count) & (
        disease_proportion_df > filter_prop
    )
    filter_disease_df.columns = ["disease_included"]

    disease_stats_df = disease_counts_df.merge(
        disease_proportion_df,
        left_index=True,
        right_index=True,
        suffixes=("_count", "_proportion"),
    ).merge(filter_disease_df, left_index=True, right_index=True)

    if not test and output_directory is not None:
        filter_file = "{}_filtered_cancertypes.tsv".format(gene)
        filter_file = os.path.join(output_directory, filter_file)
        disease_stats_df.to_csv(filter_file, sep="\t")

    if filter_cancer_types:
        use_diseases = disease_stats_df.query("disease_included").index.tolist()
        y_df = y_df.query("DISEASE in @use_diseases")

    return y_df, valid_samples


def process_y_matrix_cancertype(acronym,
                                sample_freeze,
                                mutation_burden,
                                hyper_filter=5):
    """Build a y vector based on cancer-type membership.

    Arguments
    ---------
    acronym (str): the TCGA cancer-type barcode
    sample_freeze (pd.DataFrame): stores TCGA barcodes and cancer-types
    mutation_burden (pd.DataFrame): log10 mutation count per sample
                                    (this gets added as covariate)
    hyper_filter (float): the number of std dev above log10 mutation burden
                          to filter

    Returns
    -------
    y_df: 0/1 status DataFrame for the given cancer type
    count_df: status count dataframe
    """
    y_df = sample_freeze.assign(status=0)
    y_df.loc[y_df.DISEASE == acronym, "status"] = 1

    y_df = y_df.set_index("SAMPLE_BARCODE").merge(
        mutation_burden, left_index=True, right_index=True
    )

    burden_filter = y_df["log10_mut"] < hyper_filter * y_df["log10_mut"].std()
    y_df = y_df.loc[burden_filter, :]

    count_df = pd.DataFrame(y_df.status.value_counts()).reset_index()
    count_df.columns = ["status", acronym]

    return y_df, count_df


def align_matrices(x_file_or_df,
                   y,
                   add_cancertype_covariate=True,
                   add_mutation_covariate=True,
                   add_age_covariate=False):
    """
    Process the x matrix for the given input file and align x and y together

    Arguments
    ---------
    x_file_or_df: string location of the x matrix or matrix df itself
    y: pandas DataFrame storing status of corresponding samples
    add_cancertype_covariate: if true, add one-hot encoded cancer type as a covariate
    add_mutation_covariate: if true, add log10(mutation burden) as a covariate

    Returns
    -------
    use_samples: the samples used to subset
    rnaseq_df: processed X matrix
    y_df: processed y matrix
    gene_features: real-valued gene features, to be standardized later
    """
    try:
        x_df = pd.read_csv(x_file_or_df, index_col=0, sep='\t')
    except:
        x_df = x_file_or_df

    # select samples to use, assuming y has already been filtered by cancer type
    use_samples = y.index.intersection(x_df.index)
    x_df = x_df.reindex(use_samples)
    y = y.reindex(use_samples)

    # add features to X matrix if necessary
    gene_features = np.ones(x_df.shape[1]).astype('bool')

    if add_cancertype_covariate:
        # add one-hot covariate for cancer type
        covariate_df = pd.get_dummies(y.DISEASE)
        x_df = x_df.merge(covariate_df, left_index=True, right_index=True)

    if add_mutation_covariate:
        # add covariate for mutation burden
        mutation_covariate_df = pd.DataFrame(y.loc[:, 'log10_mut'], index=y.index)
        x_df = x_df.merge(mutation_covariate_df, left_index=True, right_index=True)

    if add_age_covariate:
        # add covariate for patient age, used for survival prediction
        age_covariate_df = pd.DataFrame(y.loc[:, 'age'], index=y.index)
        x_df = x_df.merge(age_covariate_df, left_index=True, right_index=True)

    num_added_features = x_df.shape[1] - gene_features.shape[0]
    if num_added_features > 0:
        gene_features = np.concatenate(
            (gene_features, np.zeros(num_added_features).astype('bool'))
        )

    return use_samples, x_df, y, gene_features


def preprocess_data(X_train_raw_df,
                    X_test_raw_df,
                    gene_features,
                    y_train_df=None,
                    standardize_columns=True,
                    feature_selection='mad',
                    num_features=-1,
                    bc_titration_ratio=None,
                    train_batches=None,
                    test_batches=None,
                    seed=cfg.default_seed):
    """
    Data processing and feature selection, if applicable.

    Note this needs to happen for train and test sets independently.
    """
    # first do feature selection
    if feature_selection == 'mad' and num_features > 0:
        # subset to top MAD genes (if necessary)
        X_train_raw_df, X_test_raw_df, gene_features = subset_by_mad(
            X_train_raw_df, X_test_raw_df, gene_features, num_features
        )
    elif feature_selection == 'f_test' and num_features > 0:
        X_train_raw_df, X_test_raw_df, gene_features = subset_f_test(
            X_train_raw_df, X_test_raw_df, y_train_df, gene_features, num_features
        )

    # then batch correct some of the features (if necessary)
    if bc_titration_ratio is not None:
        import mpmp.utilities.batch_utilities as bu
        if cfg.bc_covariates:
            X_train_raw_df, X_test_raw_df = bu.limma_ratio(
                X_train_raw_df,
                X_test_raw_df,
                train_batches,
                test_batches,
                bc_titration_ratio,
                columns=None,
                seed=seed
            )
        else:
            X_train_raw_df, X_test_raw_df = bu.limma_ratio(
                X_train_raw_df,
                X_test_raw_df,
                train_batches,
                test_batches,
                bc_titration_ratio,
                columns=gene_features,
                seed=seed
            )
    # then standardize each column (if necessary)
    if standardize_columns:
        X_train_df = standardize_features(X_train_raw_df, gene_features)
        X_test_df = standardize_features(X_test_raw_df, gene_features)
    else:
        X_train_df = X_train_raw_df
        X_test_df = X_test_raw_df
    return X_train_df, X_test_df


def standardize_features(X_df, gene_features):
    """Standardize real-valued features."""
    std_covariates = [(c in cfg.standardize_covariates) for c in X_df.columns]
    std_features = gene_features | std_covariates
    if np.any(std_features):
        return standardize_selected_features(X_df, std_features)
    else:
        # if no features to standardize, return the original data
        return X_df


def standardize_selected_features(X_df, gene_features):
    """Standardize (take z-scores of) selected real-valued features.

    Note this should be done for train and test sets independently. Also note
    this doesn't necessarily preserve the order of features (this shouldn't
    matter in most cases).
    """
    X_df_gene = X_df.loc[:, gene_features]
    X_df_other = X_df.loc[:, ~gene_features]
    X_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_df_gene),
        index=X_df_gene.index.copy(),
        columns=X_df_gene.columns.copy()
    )
    return pd.concat((X_df_scaled, X_df_other), axis=1)


def subset_by_mad(X_train_df, X_test_df, gene_features, subset_mad_genes, verbose=False):
    """Subset features by mean absolute deviation.

    Takes the top subset_mad_genes genes (sorted in descending order),
    calculated on the training set.

    Arguments
    ---------
    X_train_df: training data, samples x genes
    X_test_df: test data, samples x genes
    gene_features: numpy bool array, indicating which features are genes
                   (and should be subsetted/standardized)
    subset_mad_genes (int): number of genes to take

    Returns
    -------
    (train_df, test_df, gene_features) datasets with filtered features
    """
    if verbose:
        print('Taking subset of gene features', file=sys.stderr)

    mad_genes_df = (
        X_train_df.loc[:, gene_features]
                  .mad(axis=0)
                  .sort_values(ascending=False)
                  .reset_index()
    )
    mad_genes_df.columns = ['gene_id', 'mean_absolute_deviation']
    mad_genes = mad_genes_df.iloc[:subset_mad_genes, :].gene_id.astype(str).values

    non_gene_features = X_train_df.columns.values[~gene_features]
    valid_features = np.concatenate((mad_genes, non_gene_features))

    gene_features = np.concatenate((
        np.ones(mad_genes.shape[0]).astype('bool'),
        np.zeros(non_gene_features.shape[0]).astype('bool')
    ))
    train_df = X_train_df.reindex(valid_features, axis='columns')
    test_df = X_test_df.reindex(valid_features, axis='columns')
    return train_df, test_df, gene_features


def subset_f_test(X_train_df,
                  X_test_df,
                  y_train_df,
                  gene_features,
                  num_features,
                  verbose=False):
    """Subset features using univariate f-test p-values with the training labels.

    Takes the top num_features features.

    Arguments
    ---------
    X_train_df: training data, samples x genes
    X_test_df: test data, samples x genes
    y_train_df: training labels, samples x 1
    gene_features: numpy bool array, indicating which features are genes
                   (and should be included in feature selection)
    num_features (int): number of features to select

    Returns
    -------
    (train_df, test_df, gene_features) datasets with filtered features
    """
    from sklearn.feature_selection import f_classif, SelectKBest

    if y_train_df is None:
        # TODO actual error handling?
        print('y_train_df cannot be none', file=sys.stderr)
        exit()

    if verbose:
        print('Performing feature selection using f-test', file=sys.stderr)

    # TODO sandbox this
    X_gene_train_df = X_train_df.loc[:, gene_features]
    X_gene_test_df = X_test_df.loc[:, gene_features]
    X_non_gene_train_df = X_train_df.loc[:, ~gene_features]
    X_non_gene_test_df = X_test_df.loc[:, ~gene_features]

    selector = SelectKBest(f_classif, k=num_features)
    selector.fit(X_gene_train_df, y_train_df)
    select_cols = selector.get_support(indices=True)

    train_df = pd.concat(
        (X_gene_train_df.iloc[:, select_cols], X_non_gene_train_df),
        axis='columns'
    )
    test_df = pd.concat(
        (X_gene_test_df.iloc[:, select_cols], X_non_gene_test_df),
        axis='columns'
    )
    gene_features = np.concatenate((
        np.ones(num_features).astype('bool'),
        np.zeros(np.count_nonzero(~gene_features)).astype('bool')
    ))
    return train_df, test_df, gene_features


def preprocess_multi_data(X_train_raw_df,
                          X_test_raw_df,
                          gene_features,
                          data_types,
                          standardize_columns=True,
                          subset_mad_genes=-1):
    """
    Data processing for multi-omics experiments.

    Arguments
    ---------
    X_train_raw_df (pd.DataFrame): training dataset
    X_test_raw_df (pd.DataFrame): test dataset
    gene_features (np.array): 1D boolean array, indicating which features
                              correspond to genes and should be preprocessed
    data_types (np.array): 1D int array, indicating which data type each
                           feature corresponds to. Non-gene features should
                           have a data type of config.NONGENE_FEATURE, and other
                           features should be indexed from 0.
    standardize_columns (list): list of whether or not to standardize each
                                data type, in the same order as indexing in
                                data_types. Defaults to True for all data
                                types.

    Returns
    -------
    X_train_df (pd.DataFrame): preprocessed training data
    X_test_df (pd.DataFrame): preprocessed test data
    """
    # standardize_columns should be a list having the same length as
    # the number of data types, in the same order
    # default to standardizing all data types
    if not isinstance(standardize_columns, list):
        standardize_columns = [True] * (np.unique(data_types).shape[0] - 1)

    # first deal with subsetting gene features by MAD
    if subset_mad_genes > 0:
        train_datasets, test_datasets = [], []
        gene_features_filtered = []
        data_types_filtered = []

        # for each gene feature dataset, separately take top n mad genes
        for data_type in np.unique(data_types):

            # skip non-gene features, we don't want to subset those
            # we can add them back untransformed at the end
            if data_type == cfg.NONGENE_FEATURE: continue

            # subset to the features for the given data type, and filter
            data_ixs = (data_types == data_type)
            X_train_data_df = X_train_raw_df.loc[:, data_ixs]
            X_test_data_df = X_test_raw_df.loc[:, data_ixs]
            X_train_data_df, X_test_data_df, _ = subset_by_mad(
                X_train_data_df,
                X_test_data_df,
                np.ones((X_train_data_df.shape[1],), dtype=bool),
                subset_mad_genes
            )

            # append info for current data type
            train_datasets.append(X_train_data_df)
            test_datasets.append(X_test_data_df)
            gene_features_filtered += [True] * X_train_data_df.shape[1]
            data_types_filtered += [data_type] * X_train_data_df.shape[1]

        # then add non-gene features, this preserves the order from before
        X_train_non_gene_df = X_train_raw_df.loc[:, ~gene_features]
        X_test_non_gene_df = X_test_raw_df.loc[:, ~gene_features]
        train_datasets.append(X_train_non_gene_df)
        test_datasets.append(X_test_non_gene_df)
        gene_features_filtered += [False] * X_train_non_gene_df.shape[1]
        data_types_filtered += [cfg.NONGENE_FEATURE] * X_train_non_gene_df.shape[1]

        # then concatenate all datasets together to get the final df
        X_train_raw_df = pd.concat(train_datasets, axis=1)
        X_test_raw_df = pd.concat(test_datasets, axis=1)
        gene_features_filtered = np.array(gene_features_filtered)
        data_types_filtered = np.array(data_types_filtered)

        assert X_train_raw_df.shape[1] == X_test_raw_df.shape[1]
        assert X_train_raw_df.shape[1] == gene_features_filtered.shape[0]
        assert X_train_raw_df.shape[1] == data_types_filtered.shape[0]

        # next, standardize columns using the new gene features
        X_train_df = standardize_multi_gene_features(
            X_train_raw_df,
            standardize_columns,
            gene_features_filtered,
            data_types_filtered
        )
        X_test_df = standardize_multi_gene_features(
            X_test_raw_df,
            standardize_columns,
            gene_features_filtered,
            data_types_filtered
        )

    else:
        # if no filtering by MAD, just standardize using original gene features
        X_train_df = standardize_multi_gene_features(
            X_train_raw_df,
            standardize_columns,
            gene_features,
            data_types
        )
        X_test_df = standardize_multi_gene_features(
            X_test_raw_df,
            standardize_columns,
            gene_features,
            data_types
        )

    return X_train_df, X_test_df


def standardize_multi_gene_features(X_df, standardize_columns, gene_features, data_types):
    """Standardize features for multiple data types.

    Functions similarly to standardize_features, but applied to each data
    type in data_types separately.
    """
    # for each gene feature dataset, take top n mad genes (separately)
    datasets = []

    # process the non-gene features last
    process_data_types = np.append(np.unique(data_types)[1:], cfg.NONGENE_FEATURE)
    for data_type in process_data_types:

        # get relevant columns of X_data_df
        data_ixs = (data_types == data_type)
        X_data_df = X_df.loc[:, data_ixs]

        # for non-gene features, standardize_features will choose the ones to
        # standardize, so just pass array of zeros
        if data_type == cfg.NONGENE_FEATURE:
            X_data_df = standardize_features(
                X_data_df, np.zeros((X_data_df.shape[1],), dtype=bool)
            )
        # if we don't want to standardize the current data type, just add it
        # to the list untransformed, otherwise standardize it
        elif standardize_columns[data_type]:
            X_data_df = standardize_features(
                X_data_df, np.ones((X_data_df.shape[1],), dtype=bool)
            )
        datasets.append(X_data_df)

    # concatenate datasets back together and return
    return pd.concat(datasets, axis=1)


def get_all_samples(use_subsampled=False):
    """Get all possible data types (that we have data for)."""
    if use_subsampled:
        data_samples = cfg.subsampled_data_types
    else:
        assert cfg.data_types.keys() == cfg.sample_infos.keys(), (
            'make sure all data types in config.data_types also have sample info')
        data_samples = cfg.sample_infos
    return data_samples


def check_all_data_types(parser, overlap_data_types, debug=False):
    """Check that all data types in overlap_data_types are valid.

    If not, throw an argparse error.
    """
    all_data_types = get_all_samples(use_subsampled=debug).keys()
    if (set(all_data_types).intersection(overlap_data_types) !=
          set(overlap_data_types)):
        parser.error(
            'overlap data types must be subset of: [{}]'.format(
                ', '.join(list(all_data_types))
            )
        )


def get_cross_data_samples(data_types=None,
                           use_subsampled=False,
                           verbose=False,
                           n_dim=None):
    """Get set of samples included in desired data modalities."""

    # only use data types in data_types list
    if data_types is not None:
        data_samples = {
            d: f for d, f in (
                get_all_samples(use_subsampled).items()
            ) if d in data_types
        }
    else:
        data_samples = get_all_samples(use_subsampled)

    # get intersection of samples in all training datasets
    # TODO: make sure sample intersections for experiments are the same as before
    valid_samples = None
    for data_type, samples_file in data_samples.items():
        # get sample IDs for the given data type/processed data file
        if verbose:
            print('Loading sample IDs for {} data'.format(data_type))

        sample_info_df = pd.read_csv(
            samples_file, sep='\t', usecols=[0], index_col=0
        )

        if valid_samples is None:
            valid_samples = sample_info_df.index
        else:
            valid_samples = sample_info_df.index.intersection(valid_samples)

    return valid_samples


def filter_to_cross_data_samples(X_df,
                                 y_df,
                                 valid_samples=None,
                                 data_types=None,
                                 use_subsampled=False,
                                 verbose=False,
                                 n_dim=None):
    """Filter dataset to samples included in all data modalities."""

    # get samples that are valid for all data types, unless they're provided
    if valid_samples is None:
        valid_samples = get_cross_data_samples(
            data_types=data_types,
            use_subsampled=use_subsampled,
            verbose=verbose,
            n_dim=n_dim
        )

    # then reindex data and labels to common sample IDs
    if verbose:
        print('Taking intersection of sample IDs...', end='')

    X_filtered_df = X_df.reindex(X_df.index.intersection(valid_samples))
    y_filtered_df = y_df.reindex(y_df.index.intersection(valid_samples))

    if verbose:
        print('done')

    return (X_filtered_df, y_filtered_df)


def get_tcga_barcode_info():
    """Map TCGA barcodes to cancer type and sample type.

    This information is pulled from the cognoma cancer-data repo:
    https://github.com/cognoma/cancer-data/
    """
    # get code -> cancer type map
    cancer_types_df = pd.read_csv(cfg.cancer_types_url,
                                  dtype='str',
                                  keep_default_na=False)
    cancertype_codes_dict = dict(zip(cancer_types_df['TSS Code'],
                                     cancer_types_df.acronym))
    # get code -> sample type map
    sample_types_df = pd.read_csv(cfg.sample_types_url,
                                  dtype='str')
    sampletype_codes_dict = dict(zip(sample_types_df.Code,
                                     sample_types_df.Definition))
    return (cancer_types_df,
            cancertype_codes_dict,
            sample_types_df,
            sampletype_codes_dict)


def get_and_save_sample_info(tcga_df,
                             sampletype_codes_dict,
                             cancertype_codes_dict,
                             training_data='expression'):
    """Extract cancer type/sample type info from TCGA samples.

    Also save info to a TSV file, to use for classification/analysis later.

    Arguments
    ---------
    tcga_df (pd.DataFrame): df with sample IDs as index
    sampletype_codes_dict (dict): maps last 2 digits of TCGA barcode to sample type
    cancertype_codes_dict (dict): maps first 2 digits of TCGA barcode to cancer type
    training_data (str): describes what type of data is being downloaded

    Returns
    -------
    tcga_id (pd.DataFrame): df describing sample type/cancer type for included samples
    """

    # extract sample type in the order of the gene expression matrix
    tcga_id = pd.DataFrame(tcga_df.index)

    # extract the last two digits of the barcode and recode sample-type
    tcga_id = tcga_id.assign(sample_type = tcga_id.sample_id.str[-2:])
    tcga_id.sample_type = tcga_id.sample_type.replace(sampletype_codes_dict)

    # extract the first two ID numbers after `TCGA-` and recode cancer-type
    tcga_id = tcga_id.assign(
        cancer_type=tcga_id.sample_id.str.split('TCGA-', expand=True)[1].str[:2]
     )
    tcga_id.cancer_type = tcga_id.cancer_type.replace(cancertype_codes_dict)

    # append cancer-type with sample-type to generate stratification variable
    tcga_id = tcga_id.assign(id_for_stratification = tcga_id.cancer_type.str.cat(tcga_id.sample_type))

    # get stratification counts - function cannot work with singleton strats
    stratify_counts = tcga_id.id_for_stratification.value_counts().to_dict()

    # recode stratification variables if they are singletons
    tcga_id = tcga_id.assign(stratify_samples_count = tcga_id.id_for_stratification)
    tcga_id.stratify_samples_count = tcga_id.stratify_samples_count.replace(stratify_counts)
    tcga_id.loc[tcga_id.stratify_samples_count == 1, "stratify_samples"] = "other"

    # write files for downstream use
    os.makedirs(cfg.sample_info_dir, exist_ok=True)
    fname = os.path.join(cfg.sample_info_dir,
                         'tcga_{}_sample_identifiers.tsv'.format(training_data))

    tcga_id.drop(['stratify_samples', 'stratify_samples_count'], axis='columns', inplace=True)
    tcga_id.to_csv(fname, sep='\t', index=False)

    return tcga_id


def get_compress_output_prefix(data_type,
                               n_dim,
                               seed,
                               standardize_input=True):
    output_prefix = '{}_pc{}_s{}'.format(data_type, n_dim, seed)
    if standardize_input:
        output_prefix += '_std'
    return output_prefix


def compress_and_save_data(data_type,
                           input_data_df,
                           output_dir,
                           n_dim,
                           standardize_input,
                           verbose=False,
                           seed=cfg.default_seed,
                           save_variance_explained=False):

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n, p = input_data_df.shape
    if n_dim > min(n, p):
        if verbose:
            print('n_dim > min(n, p), so setting n_dim = min(n, p)')
        n_dim = min(n, p)

    output_prefix = get_compress_output_prefix(data_type,
                                               n_dim,
                                               seed,
                                               standardize_input)

    # standardize first for some data types
    if standardize_input:
        input_data_df = pd.DataFrame(
            StandardScaler().fit_transform(input_data_df),
            index=input_data_df.index.copy(),
            columns=input_data_df.columns.copy()
        )

    # calculate PCA and save compressed data matrix
    pca = PCA(n_components=n_dim, random_state=seed)
    transformed_data_df = pd.DataFrame(
        pca.fit_transform(input_data_df),
        index=input_data_df.index,
        columns=['PC{}'.format(i) for i in range(n_dim)]
    )
    transformed_data_df.to_csv(
        os.path.join(output_dir, '{}.tsv.gz'.format(output_prefix)),
        sep='\t', float_format='%.3g')

    if save_variance_explained:
        np.savetxt(os.path.join(output_dir, '{}_ve.tsv.gz'.format(output_prefix)),
                   pca.explained_variance_ratio_)

    return transformed_data_df



