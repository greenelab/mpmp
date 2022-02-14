import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du
from mpmp.utilities.tcga_utilities import (
    process_y_matrix,
    process_y_matrix_cancertype,
    align_matrices,
    filter_to_cross_data_samples,
)

class TCGADataModel():
    """
    Class containing data necessary to run TCGA mutation prediction experiments.

    Provides an interface to load and preprocess mutation data and training data
    modalities, and to split data into train/test sets for each target gene.
    """

    def __init__(self,
                 seed=cfg.default_seed,
                 training_data='expression',
                 overlap_data_types=None,
                 load_compressed_data=False,
                 standardize_input=False,
                 n_dim=None,
                 sample_info_df=None,
                 verbose=False,
                 debug=False,
                 test=False):
        """
        Initialize mutation prediction model/data

        Arguments
        ---------
        seed (int): seed for random number generator
        training_data (str): what data type to train the model on
        overlap_data_types (list): what data types to use to determine sample set
        load_compressed_data (bool): whether or not to use compressed data
        n_dim (int): how many dimensions to use for compression algorithm
        verbose (bool): whether or not to write verbose output
        sample_info_df (pd.DataFrame): dataframe containing info about TCGA samples
        debug (bool): if True, use a subset of expression data for quick debugging
        test (bool): if True, don't save results to files
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.compressed_data = load_compressed_data
        self.overlap_data_types = overlap_data_types
        self.n_dim = n_dim
        self.verbose = verbose
        self.debug = debug
        self.test = test

        # load and store data in memory
        self._load_data(train_data_type=training_data,
                        compressed_data=load_compressed_data,
                        standardize_input=standardize_input,
                        n_dim=n_dim,
                        sample_info_df=sample_info_df,
                        debug=debug,
                        test=self.test)

    def load_gene_set(self, gene_set='top_50'):
        """
        Load gene set data from previous GitHub repos.

        Arguments
        ---------
        gene_set (str): which predefined gene set to use, or a list of gene names
                        to use a custom list.

        Returns
        -------
        genes_df (pd.DataFrame): list of genes to run cross-validation experiments for,
                                 contains gene names and oncogene/TSG classifications
        """
        if self.verbose:
            print('Loading gene label data...', file=sys.stderr)

        if gene_set == 'top_50':
            genes_df = du.load_top_genes()
        elif gene_set == 'vogelstein':
            genes_df = du.load_vogelstein()
        elif gene_set == '50_random':
            genes_df = du.load_random_genes()
        else:
            from mpmp.exceptions import GenesNotFoundError
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_vogelstein()
            # if all genes in gene_set are in vogelstein dataset, use it
            if set(gene_set).issubset(set(genes_df.gene.values)):
                genes_df = genes_df[genes_df.gene.isin(gene_set)]
            # else if all genes in gene_set are in top50 dataset, use it
            else:
                genes_df = du.load_top_genes()
                if set(gene_set).issubset(set(genes_df.gene.values)):
                    genes_df = genes_df[genes_df.gene.isin(gene_set)]
                else:
                    # else if all genes in gene_set are in random dataset, use it
                    genes_df = du.load_random_genes()
                    if set(gene_set).issubset(set(genes_df.gene.values)):
                        genes_df = genes_df[genes_df.gene.isin(gene_set)]
                    else:
                        # else, finally, throw an error
                        raise GenesNotFoundError(
                            'Gene list was not a subset of existing gene sets'
                        )

        return genes_df

    def process_data_for_cancer_type(self,
                                     cancer_type,
                                     cancer_type_dir):
        """
        Prepare to run cancer type prediction experiments.

        This has to be rerun to generate the labels for each cancer type.

        Arguments
        ---------
        cancer_type (str): cancer type to predict (one vs. rest binary)
        cancer_type_dir (str): directory to write output to, if None don't
                               write output
        """
        y_df_raw = self._generate_cancer_type_labels(cancer_type)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
            train_filtered_df,
            y_filtered_df,
            data_types=self.overlap_data_types,
            use_subsampled=(self.debug or self.test),
            verbose=self.verbose
        )

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def process_data_for_gene(self,
                              gene,
                              classification,
                              gene_dir,
                              use_pancancer=False,
                              filter_cancer_types=True,
                              batch_correction=False,
                              bc_cancer_type=False):
        """
        Prepare to run mutation prediction experiments for a given gene.

        Arguments
        ---------
        gene (str): gene to run experiments for
        classification (str): 'oncogene' or 'TSG'; most likely cancer function for
                              the given gene
        gene_dir (str): directory to write output to, if None don't write output
        use_pancancer (bool): whether or not to use pancancer data
        """
        y_df_raw, valid_samples = self._generate_gene_labels(
                gene, classification, gene_dir, filter_cancer_types)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw,
            add_cancertype_covariate=True
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        # add non-gene features to data_types array if necessary
        # this is used when building multi-omics models
        if hasattr(self, 'data_types'):
            # this has to have a different name than the general data_types
            # array, since this preprocessing may happen multiple times (for
            # each gene) in the same script call
            self.gene_data_types = np.concatenate(
                (self.data_types, np.array([cfg.NONGENE_FEATURE] *
                                            np.count_nonzero(~gene_features)))
            )
            assert self.gene_data_types.shape[0] == gene_features.shape[0]

        train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
            train_filtered_df,
            y_filtered_df,
            valid_samples=valid_samples,
            data_types=self.overlap_data_types,
            n_dim=self.n_dim,
            use_subsampled=(self.debug or self.test),
            verbose=self.verbose
        )

        if batch_correction:
            import mpmp.utilities.batch_utilities as bu
            # we're using the mutation as a batch indicator
            # this will effectively remove all linear signal in the dataset
            # related to presence/absence of the mutation
            if cfg.bc_covariates:
                train_filtered_df, _ = bu.run_limma(
                    train_filtered_df,
                    y_filtered_df.status.astype(str).values,
                    verbose=self.verbose)
            else:
                train_filtered_df, _ = bu.run_limma(
                    train_filtered_df,
                    y_filtered_df.status.astype(str).values,
                    columns=gene_features,
                    verbose=self.verbose)
        elif bc_cancer_type:
            import mpmp.utilities.batch_utilities as bu
            # we're using the cancer type as a batch indicator
            cancer_type_to_index = {
                ct: ix for ix, ct in enumerate(y_filtered_df.DISEASE.unique())
            }
            train_filtered_df = bu.run_limma(
                train_filtered_df,
                np.array([cancer_type_to_index[ct] for ct in y_filtered_df.DISEASE]),
                verbose=self.verbose)

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_purity_data(self,
                            output_dir,
                            classify=False):
        """Prepare to run experiments predicting tumor purity.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        classify (bool): if True do classification, else regression
        """
        y_df_raw = du.load_purity(self.mut_burden_df,
                                  self.sample_info_df,
                                  classify=classify,
                                  verbose=self.verbose)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw,
            add_cancertype_covariate=True
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
            train_filtered_df,
            y_filtered_df,
            data_types=self.overlap_data_types,
            n_dim=self.n_dim,
            use_subsampled=(self.debug or self.test),
            verbose=self.verbose
        )

        # filter to samples in common between training data and tumor purity
        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_msi_data(self, cancer_type, output_dir):
        """Prepare to run experiments predicting microsatellite instability status.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        classify (bool): if True do classification, else regression
        """
        y_df_raw = du.load_msi(cancer_type,
                               self.mut_burden_df,
                               self.sample_info_df,
                               verbose=self.verbose)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw,
            add_cancertype_covariate=(cancer_type == 'pancancer')
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
            train_filtered_df,
            y_filtered_df,
            data_types=self.overlap_data_types,
            n_dim=self.n_dim,
            use_subsampled=(self.debug or self.test),
            verbose=self.verbose
        )

        # filter to samples in common between training data and tumor purity
        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_survival_data(self,
                              output_dir,
                              cancer_type):
        """Prepare to run experiments predicting survival from omics data.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        """
        y_df_raw = du.load_survival_labels(cancer_type,
                                           self.mut_burden_df,
                                           self.sample_info_df,
                                           verbose=self.verbose)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw,
            # add cancer type covariate only in pan-cancer prediction case
            add_cancertype_covariate=(cancer_type == 'pancancer'),
            add_age_covariate=True
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
            train_filtered_df,
            y_filtered_df,
            data_types=self.overlap_data_types,
            n_dim=self.n_dim,
            use_subsampled=(self.debug or self.test),
            verbose=self.verbose
        )

        # filter to samples in common between training data and tumor purity
        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def _load_data(self,
                   train_data_type,
                   compressed_data=False,
                   standardize_input=False,
                   n_dim=None,
                   sample_info_df=None,
                   debug=False,
                   test=False):
        """Load and store relevant data.

        This data does not vary based on the gene/cancer type being considered
        (i.e. it can be loaded only once when the class is instantiated).

        Arguments:
        ----------
        debug (bool): whether or not to subset data for faster debugging
        test (bool): whether or not to subset columns in mutation data, for testing
        """
        # first load and unpack pancancer mutation/CNV/TMB data
        # this data is described in more detail in the load_pancancer_data docstring
        if test:
            # for testing, just load a subset of pancancer data,
            # this is much faster than loading mutation data for all genes
            import mpmp.test_config as tcfg
            pancan_data = du.load_pancancer_data(verbose=self.verbose,
                                                 test=True,
                                                 subset_columns=tcfg.test_genes)
        else:
            pancan_data = du.load_pancancer_data(verbose=self.verbose)

        (self.sample_freeze_df,
         self.mutation_df,
         self.copy_loss_df,
         self.copy_gain_df,
         self.mut_burden_df) = pancan_data

        # now load training data
        if not isinstance(train_data_type, str):
            # if a list of train data types is provided, we have to load each
            # of them and concatenate columns
            # n_dim should be a list here
            self.data_df, self.data_types = du.load_multiple_data_types(
                                                train_data_type,
                                                n_dims=n_dim,
                                                standardize_input=standardize_input,
                                                verbose=self.verbose)
        elif compressed_data:
            self.data_df = du.load_compressed_data(train_data_type,
                                                   n_dim=n_dim,
                                                   verbose=self.verbose,
                                                   standardize_input=standardize_input,
                                                   load_subset=(debug or test))
        elif train_data_type == 'baseline':
            # we just want to use non-omics covariates as a baseline
            # so here, get sample list for expression data, then create an
            # empty data frame using it as an index
            if sample_info_df is None:
                sample_info_df = du.load_sample_info('expression',
                                                     verbose=self.verbose)
            self.data_df = pd.DataFrame(index=sample_info_df.index)
        else:
            if train_data_type == 'vogelstein_mutations':
                self.data_df = self._load_vogelstein_mutation_matrix()
            elif train_data_type == 'significant_mutations':
                data_df = self._load_vogelstein_mutation_matrix()
                sig_genes = du.load_significant_genes('methylation')
                # startswith() with a tuple argument returns True if
                # the string matches any of the prefixes in the tuple
                # https://stackoverflow.com/a/20461857
                self.data_df = data_df.loc[
                    :, data_df.columns.str.startswith(tuple(sig_genes))
                ]
            elif 'mutation_preds' in train_data_type:
                self.data_df = du.load_mutation_predictions(train_data_type)
            else:
                self.data_df = du.load_raw_data(train_data_type,
                                                verbose=self.verbose,
                                                load_subset=(debug or test))

        if sample_info_df is None:
            self.sample_info_df = du.load_sample_info(train_data_type,
                                                      verbose=self.verbose)
        else:
            # sometimes we load sample info in the calling script as part of
            # argument processing, etc
            # in that case, we don't need to load it again
            self.sample_info_df = sample_info_df


    def _generate_cancer_type_labels(self, cancer_type):
        y_df, count_df = process_y_matrix_cancertype(
            acronym=cancer_type,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            hyper_filter=5,
        )
        return y_df

    def _generate_gene_labels(self,
                              gene,
                              classification,
                              gene_dir,
                              filter_cancer_types=True):

        # process the y matrix for the given gene or pathway
        y_mutation_df = self.mutation_df.loc[:, gene]

        # include copy number gains for oncogenes
        # and copy number loss for tumor suppressor genes (TSG)
        include_copy = True
        if classification == "Oncogene":
            y_copy_number_df = self.copy_gain_df.loc[:, gene]
        elif classification == "TSG":
            y_copy_number_df = self.copy_loss_df.loc[:, gene]
        elif classification == "Oncogene, TSG":
            # some genes may act as both (i.e. in a cancer type-specific
            # or tissue-specific manner), in this case we'll just use the
            # union of the gain/loss dfs to define positive labeled samples
            y_copy_number_df = (
                self.copy_gain_df.loc[:, gene] | self.copy_loss_df.loc[:, gene]
            )
        else:
            y_copy_number_df = pd.DataFrame()
            include_copy = False

        # construct labels from mutation/CNV information, and filter for
        # cancer types without an extreme label imbalance
        y_df, valid_samples = process_y_matrix(
            y_mutation=y_mutation_df,
            y_copy=y_copy_number_df,
            include_copy=include_copy,
            gene=gene,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            filter_cancer_types=filter_cancer_types,
            filter_count=cfg.filter_count,
            filter_prop=cfg.filter_prop,
            output_directory=gene_dir,
            hyper_filter=5,
            test=self.test,
            overlap_data_types=self.overlap_data_types
        )
        return y_df, valid_samples

    def _filter_data(self,
                     data_df,
                     y_df,
                     add_cancertype_covariate=False,
                     add_age_covariate=False):
        use_samples, data_df, y_df, gene_features = align_matrices(
            x_file_or_df=data_df,
            y=y_df,
            add_cancertype_covariate=add_cancertype_covariate,
            add_mutation_covariate=True,
            add_age_covariate=add_age_covariate
        )
        return data_df, y_df, gene_features

    def _load_vogelstein_mutation_matrix(self):
        """Load mutation info for all Vogelstein genes, to be used as predictors."""

        if self.verbose:
            print('Loading Vogelstein mutation matrix...', file=sys.stderr)

        vogelstein_mutation_df = []
        sample_ids = None

        genes_df = du.load_vogelstein()
        for gene_idx, gene_series in genes_df.iterrows():

            gene = gene_series.gene
            classification = gene_series.classification

            # construct labels from mutation/CNV information
            # crucially, we *do not* want to filter cancer types with extreme
            # label imbalance here, since we want to have a square matrix of
            # predictors (i.e. data for all cancer types in survival dataset)
            try:
                gene_labels, _ = self._generate_gene_labels(
                    gene, classification, gene_dir=None, filter_cancer_types=False)
                if sample_ids is None:
                    sample_ids = gene_labels.index
                else:
                    assert sample_ids.equals(gene_labels.index)
                vogelstein_mutation_df.append(
                    [gene + '_status'] + list(gene_labels.status.values)
                )

            except KeyError:
                # just skip genes that don't have mutation data
                continue

        vogelstein_mutation_df = (
            pd.DataFrame(vogelstein_mutation_df,
                         columns=['gene'] + list(sample_ids.values))
              .transpose()
        )

        vogelstein_mutation_df = (vogelstein_mutation_df
            .rename(columns=vogelstein_mutation_df.iloc[0])
            .drop(vogelstein_mutation_df.index[0])
            # columns will have object type, we want to convert to int
            .astype('uint8')
        )

        return vogelstein_mutation_df


