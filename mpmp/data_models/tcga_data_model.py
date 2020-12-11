import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd

import mpmp.config as cfg
import mpmp.utilities.data_utilities as du
from mpmp.utilities.tcga_utilities import (
    process_y_matrix_cancertype,
    align_matrices,
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
        verbose (bool): whether or not to write verbose output
        debug (bool): if True, use a subset of expression data for quick debugging
        test (bool): if True, don't save results to files
        """
        # save relevant parameters
        np.random.seed(seed)
        self.seed = seed
        self.subset_mad_genes = subset_mad_genes
        self.verbose = verbose
        self.test = test

        # load and store data in memory
        self._load_data(train_data_type=training_data,
                        sample_info_df=sample_info_df,
                        debug=debug,
                        test=self.test)

    def load_gene_set(self, gene_set='top_50'):
        """
        Load gene set data from previous GitHub repos.

        Arguments
        ---------
        gene_set (str): which predefined gene set to use, or 'custom' for custom list.

        Returns
        -------
        genes_df (pd.DataFrame): list of genes to run cross-validation experiments for,
                                 contains gene names and oncogene/TSG classifications

        TODO: still not sure how to generalize oncogene/TSG info past these
        predefined gene sets, should eventually look into how to do this
        """
        if self.verbose:
            print('Loading gene label data...', file=sys.stderr)

        if gene_set == 'top_50':
            genes_df = du.load_top_50()
        elif gene_set == 'vogelstein':
            genes_df = du.load_vogelstein()
        else:
            assert isinstance(gene_set, typing.List)
            genes_df = du.load_vogelstein()
            if gene in genes_df.gene:
                genes_df = genes_df[genes_df.gene.isin(gene_set)]
            else:
                genes_df = load_top_50()
                genes_df = genes_df[genes_df.gene.isin(gene_set)]

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

        filtered_data = self._filter_data_for_cancer_type(self.train_df,
                                                          y_df_raw)

        train_filtered_df, y_filtered_df, gene_features = filtered_data

        if shuffle_labels:
            y_filtered_df.status = np.random.permutation(
                y_filtered_df.status.values)

        self.X_df = train_filtered_df
        self.y_df = y_filtered_df
        self.gene_features = gene_features

    def _load_data(self,
                   train_data_type,
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
        if train_data_type == 'expression':
            self.train_df = du.load_expression_data(verbose=self.verbose,
                                                    debug=debug)
        elif train_data_type == 'methylation':
            self.train_df = du.load_methylation_data(verbose=self.verbose,
                                                     debug=debug)

        self.sample_info_df = du.load_sample_info(verbose=self.verbose)

        # load and unpack pancancer data
        # this data is described in more detail in the load_pancancer_data docstring
        if test:
            # for testing, just load a subset of pancancer data,
            # this is much faster than loading mutation data for all genes
            pancan_data = du.load_pancancer_data(verbose=self.verbose,
                                                 test=True,
                                                 subset_columns=cfg.test_genes)
        else:
            pancan_data = du.load_pancancer_data(verbose=self.verbose)

        (self.sample_freeze_df,
         self.mutation_df,
         self.copy_loss_df,
         self.copy_gain_df,
         self.mut_burden_df) = pancan_data

    def _generate_cancer_type_labels(self, cancer_type):
        # TODO: should we do something with cancer type counts?
        y_df, count_df = process_y_matrix_cancertype(
            acronym=cancer_type,
            sample_freeze=self.sample_freeze_df,
            mutation_burden=self.mut_burden_df,
            hyper_filter=5,
        )
        return y_df

    def _filter_data_for_cancer_type(self, train_df, y_df):
        use_samples, train_df, y_df, gene_features = align_matrices(
            x_file_or_df=train_df,
            y=y_df,
            add_cancertype_covariate=False,
            add_mutation_covariate=True
        )
        return train_df, y_df, gene_features


