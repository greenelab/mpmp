"""
Functions for training classifiers on TCGA data.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import (
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
#     average_precision_score
# )
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import GridSearchCV

import mpmp.config as cfg
# import mpmp.utilities.tcga_utilities as tu
from mpmp.exceptions import (
    NoTrainSamplesError,
    NoTestSamplesError,
)

def run_cv_cancer_type(data_model,
                       cancer_type,
                       sample_info,
                       num_folds,
                       shuffle_labels):
    """
    Run cancer type cross-validation experiments for a given gene, then
    write the results to files in the results directory. If the relevant
    files already exist, skip this experiment.

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    cancer_type (str): cancer type to run experiments for
    sample_info (pd.DataFrame): df with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    """
    results = {
        'gene_metrics': [],
        'gene_auc': [],
        'gene_aupr': [],
        'gene_coef': []
    }
    signal = 'shuffled' if shuffle_labels else 'signal'

    for fold_no in range(num_folds):
        X_train_raw_df, X_test_raw_df, _ = split_stratified(
           data_model.X_df, sample_info, num_folds=num_folds,
           fold_no=fold_no, seed=data_model.seed)
        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)
        print(X_train_raw_df.shape)
        print(y_train_df.shape)
        print(X_test_raw_df.shape)
        print(y_test_df.shape)
        exit()

    #    X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df, X_test_raw_df,
    #                                               data_model.gene_features,
    #                                               data_model.subset_mad_genes)

    # try:
    #     # also ignore warnings here, same deal as above
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         model_results = train_model(
    #             X_train=X_train_df,
    #             X_test=X_test_df,
    #             y_train=y_train_df,
    #             alphas=cfg.alphas,
    #             l1_ratios=cfg.l1_ratios,
    #             seed=data_model.seed,
    #             n_folds=cfg.folds,
    #             max_iter=cfg.max_iter
    #         )
    #         (cv_pipeline,
    #          y_pred_train_df,
    #          y_pred_test_df,
    #          y_cv_df) = model_results
    # except ValueError:
    #     raise OneClassError(
    #         'Only one class present in test set for gene: {}\n'.format(gene)
    #     )

    # # TODO: separate below into another function (one returns raw results)

    # # get coefficients
    # coef_df = extract_coefficients(
    #     cv_pipeline=cv_pipeline,
    #     feature_names=X_train_df.columns,
    #     signal=signal,
    #     seed=data_model.seed
    # )
    # coef_df = coef_df.assign(gene=gene)
    # coef_df = coef_df.assign(fold=fold_no)

    # try:
    #     # also ignore warnings here, same deal as above
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         metric_df, gene_auc_df, gene_aupr_df = get_metrics(
    #             y_train_df, y_test_df, y_cv_df, y_pred_train_df,
    #             y_pred_test_df, gene, 'N/A', signal, data_model.seed,
    #             fold_no
    #         )
    # except ValueError:
    #     raise OneClassError(
    #         'Only one class present in test set for gene: {}\n'.format(gene)
    #     )

    # results['gene_metrics'].append(metric_df)
    # results['gene_auc'].append(gene_auc_df)
    # results['gene_aupr'].append(gene_aupr_df)
    # results['gene_coef'].append(coef_df)

    return results


def split_stratified(data_df,
                     sample_info_df,
                     num_folds=4,
                     fold_no=1,
                     seed=cfg.default_seed):
    """Split expression data into train and test sets.

    The train and test sets will both contain data from all cancer types,
    in roughly equal proportions.

    Arguments
    ---------
    data_df (pd.DataFrame): samples x features dataframe
    sample_info_df (pd.DataFrame): maps samples to cancer types
    num_folds (int): number of cross-validation folds
    fold_no (int): cross-validation fold to hold out
    seed (int): seed for deterministic splits

    Returns
    -------
    train_df (pd.DataFrame): samples x features train data
    test_df (pd.DataFrame): samples x features test data
    """
    # subset sample info to samples in pre-filtered expression data
    sample_info_df = sample_info_df.reindex(data_df.index)

    # generate id for stratification
    # this is a concatenation of cancer type and sample/tumor type, since we want
    # to stratify by both
    sample_info_df = sample_info_df.assign(
        id_for_stratification = sample_info_df.cancer_type.str.cat(
                                                sample_info_df.sample_type)
    )
    # recode stratification id if they are singletons or near-singletons,
    # since these won't work with StratifiedKFold
    stratify_counts = sample_info_df.id_for_stratification.value_counts().to_dict()
    sample_info_df = sample_info_df.assign(
        stratify_samples_count = sample_info_df.id_for_stratification
    )
    sample_info_df.stratify_samples_count = sample_info_df.stratify_samples_count.replace(
        stratify_counts)
    sample_info_df.loc[
        sample_info_df.stratify_samples_count < num_folds, 'id_for_stratification'
    ] = 'other'

    # now do stratified CV splitting and return the desired fold
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_ixs, test_ixs) in enumerate(
            kf.split(data_df, sample_info_df.id_for_stratification)):
        if fold == fold_no:
            train_df = data_df.iloc[train_ixs]
            test_df = data_df.iloc[test_ixs]
    return train_df, test_df, sample_info_df
