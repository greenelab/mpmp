"""
Functions for training classifiers on TCGA data.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
)

import mpmp.config as cfg
import mpmp.utilities.tcga_utilities as tu
from mpmp.exceptions import (
    NoTrainSamplesError,
    NoTestSamplesError,
)

def run_cv_cancer_type(data_model,
                       cancer_type,
                       sample_info,
                       num_folds,
                       shuffle_labels=False,
                       standardize_columns=False):
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
    standardize_columns (bool): whether or not to standardize predictors
    """
    results = {
        'gene_metrics': [],
        'gene_auc': [],
        'gene_aupr': [],
        'gene_coef': []
    }
    signal = 'shuffled' if shuffle_labels else 'signal'

    for fold_no in range(num_folds):
        # TODO: catch user warning?
        X_train_raw_df, X_test_raw_df, _ = split_stratified(
           data_model.X_df, sample_info, num_folds=num_folds,
           fold_no=fold_no, seed=data_model.seed)
        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df,
                                                   X_test_raw_df,
                                                   data_model.gene_features,
                                                   standardize_columns,
                                                   data_model.subset_mad_genes)

        model_results = train_model(
            X_train=X_train_df,
            X_test=X_test_df,
            y_train=y_train_df,
            alphas=cfg.alphas,
            l1_ratios=cfg.l1_ratios,
            seed=data_model.seed,
            n_folds=cfg.folds,
            max_iter=cfg.max_iter
        )
        (cv_pipeline,
         y_pred_train_df,
         y_pred_test_df,
         y_cv_df) = model_results
        print(y_pred_train_df[:5])

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


def train_model(X_train,
                X_test,
                y_train,
                alphas,
                l1_ratios,
                seed,
                n_folds=4,
                max_iter=1000):
    """
    Build the logic and sklearn pipelines to train x matrix based on input y

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix (output from align_matrices())
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """
    # Setup the classifier parameters
    clf_parameters = {
        "classify__loss": ["log"],
        "classify__penalty": ["elasticnet"],
        "classify__alpha": alphas,
        "classify__l1_ratio": l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                "classify",
                SGDClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    loss="log",
                    max_iter=max_iter,
                    tol=1e-3,
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring="roc_auc",
        return_train_score=True,
        iid=False
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train.status)

    # Obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=X_train,
        y=y_train.status,
        cv=n_folds,
        method="decision_function",
    )

    # Get all performance results
    y_predict_train = cv_pipeline.decision_function(X_train)
    y_predict_test = cv_pipeline.decision_function(X_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


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
