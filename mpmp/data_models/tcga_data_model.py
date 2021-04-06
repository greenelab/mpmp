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
                 subset_mad_genes=-1,
                 training_data='expression',
                 load_compressed_data=False,
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
        subset_mad_genes (int): how many genes to keep (top by mean absolute deviation).
                                -1 doesn't do any filtering (all genes will be kept).
        training_data (str): what data type to train the model on
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
        self.subset_mad_genes = subset_mad_genes
        self.compressed_data = load_compressed_data
        self.n_dim = n_dim
        self.verbose = verbose
        self.debug = debug
        self.test = test

        # load and store data in memory
        self._load_data(train_data_type=training_data,
                        compressed_data=load_compressed_data,
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
            genes_df = du.load_top_50()
        elif gene_set == 'vogelstein':
            genes_df = du.load_vogelstein()
        else:
            from mpmp.exceptions import GenesNotFoundError
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_vogelstein()
            # if all genes in gene_set are in vogelstein dataset, use it
            if set(gene_set).issubset(set(genes_df.gene.values)):
                genes_df = genes_df[genes_df.gene.isin(gene_set)]
            # else if all genes in gene_set are in top50 dataset, use it
            else:
                genes_df = du.load_top_50()
                if set(gene_set).issubset(set(genes_df.gene.values)):
                    genes_df = genes_df[genes_df.gene.isin(gene_set)]
                else:
                # else throw an error
                    raise GenesNotFoundError(
                        'Gene list was not a subset of Vogelstein or top50'
                    )

        return genes_df

    def process_data_for_cancer_type(self,
                                     cancer_type,
                                     cancer_type_dir,
                                     shuffle_labels=False):
        """
        Prepare to run cancer type prediction experiments.

        This has to be rerun to generate the labels for each cancer type.

        Arguments
        ---------
        cancer_type (str): cancer type to predict (one vs. rest binary)
        cancer_type_dir (str): directory to write output to, if None don't
                               write output
        shuffle_labels (bool): whether or not to shuffle labels (negative
                               control)
        """
        y_df_raw = self._generate_cancer_type_labels(cancer_type)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        if cfg.use_only_cross_data_samples:
            train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
                train_filtered_df,
                y_filtered_df,
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
                              shuffle_labels=False,
                              compressed_only=False):
        """
        Prepare to run mutation prediction experiments for a given gene.

        Arguments
        ---------
        gene (str): gene to run experiments for
        classification (str): 'oncogene' or 'TSG'; most likely cancer function for
                              the given gene
        gene_dir (str): directory to write output to, if None don't write output
        use_pancancer (bool): whether or not to use pancancer data
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        """
        y_df_raw = self._generate_gene_labels(gene, classification, gene_dir)

        filtered_data = self._filter_data(
            self.data_df,
            y_df_raw,
            add_cancertype_covariate=True
        )
        train_filtered_df, y_filtered_df, gene_features = filtered_data

        # add non-gene features to data_types array if necessary
        if hasattr(self, 'data_types'):
            # this has to have a different name than the general data_types
            # array, since this preprocessing may happen multiple times (for
            # each gene) in the same script call
            self.gene_data_types = np.concatenate(
                (self.data_types, np.array([cfg.NONGENE_FEATURE] *
                                            np.count_nonzero(~gene_features)))
            )
            assert self.gene_data_types.shape[0] == gene_features.shape[0]

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        if cfg.use_only_cross_data_samples:
            train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
                train_filtered_df,
                y_filtered_df,
                # if this option is True, use only samples for which we have
                # compressed data. if false, take overlap of samples for which
                # we have non-compressed data (generally a subset of compressed
                # data samples)
                compressed_data_only=compressed_only,
                n_dim=self.n_dim,
                use_subsampled=(self.debug or self.test),
                verbose=self.verbose
            )

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

        assert np.count_nonzero(self.X_df.index.duplicated()) == 0
        assert np.count_nonzero(self.y_df.index.duplicated()) == 0

    def process_purity_data(self,
                            output_dir,
                            classify=False,
                            shuffle_labels=False,
                            compressed_only=False):
        """Prepare to run experiments predicting tumor purity.

        Arguments
        ---------
        output_dir (str): directory to write output to, if None don't write output
        classify (bool): if True do classification, else regression
        shuffle_labels (bool): whether or not to shuffle labels (negative control)
        compressed_only (bool): if True, use intersection of compressed samples
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

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        if cfg.use_only_cross_data_samples:
            train_filtered_df, y_filtered_df = filter_to_cross_data_samples(
                train_filtered_df,
                y_filtered_df,
                # if this option is True, use only samples for which we have
                # compressed data. if False, take overlap of samples for which
                # we have non-compressed data (generally a subset of compressed
                # data samples)
                compressed_data_only=compressed_only,
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
        # load training data
        if not isinstance(train_data_type, str):
            # if a list of train data types is provided, we have to load each
            # of them and concatenate columns
            # n_dim should be a list here
            self.data_df, self.data_types = du.load_multiple_data_types(
                                                train_data_type,
                                                n_dims=n_dim,
                                                verbose=self.verbose)
        elif compressed_data:
            self.data_df = du.load_compressed_data(train_data_type,
                                                   n_dim=n_dim,
                                                   verbose=self.verbose,
                                                   load_subset=(debug or test))
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

        # load and unpack pancancer mutation/CNV/TMB data
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

    def _generate_cancer_type_labels(self, cancer_type):
        y_df, count_df = process_y_matrix_cancertype(
            acronym=cancer_type,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            hyper_filter=5,
        )
        return y_df

    def _generate_gene_labels(self, gene, classification, gene_dir):
        # process the y matrix for the given gene or pathway
        y_mutation_df = self.mutation_df.loc[:, gene]

        # include copy number gains for oncogenes
        # and copy number loss for tumor suppressor genes (TSG)
        include_copy = True
        if classification == "Oncogene":
            y_copy_number_df = self.copy_gain_df.loc[:, gene]
        elif classification == "TSG":
            y_copy_number_df = self.copy_loss_df.loc[:, gene]
        else:
            y_copy_number_df = pd.DataFrame()
            include_copy = False

        # construct labels from mutation/CNV information, and filter for
        # cancer types without an extreme label imbalance
        y_df = process_y_matrix(
            y_mutation=y_mutation_df,
            y_copy=y_copy_number_df,
            include_copy=include_copy,
            gene=gene,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            filter_count=cfg.filter_count,
            filter_prop=cfg.filter_prop,
            output_directory=gene_dir,
            hyper_filter=5,
            test=self.test
        )
        return y_df

    def _filter_data(self,
                     data_df,
                     y_df,
                     add_cancertype_covariate=False):
        use_samples, data_df, y_df, gene_features = align_matrices(
            x_file_or_df=data_df,
            y=y_df,
            add_cancertype_covariate=add_cancertype_covariate,
            add_mutation_covariate=True
        )
        return data_df, y_df, gene_features


