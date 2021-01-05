"""
Functions for training classifiers on TCGA data.

Many of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import warnings

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
    OneClassError,
)

def run_cv_stratified(data_model,
                      exp_string,
                      identifier,
                      training_data,
                      sample_info,
                      num_folds,
                      shuffle_labels=False,
                      standardize_columns=False):
    """
    Run stratified cross-validation experiments for a given dataset, then write
    the results to files in the results directory. If the relevant files already
    exist, skip this experiment.

    Arguments
    ---------
    data_model (TCGADataModel): class containing preprocessed train/test data
    exp_string (str): string describing the experiment being run
    identifier (str): string describing the target value/environment
    training_data (str): what type of data is being used to train model
    sample_info (pd.DataFrame): df with TCGA sample information
    num_folds (int): number of cross-validation folds to run
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    standardize_columns (bool): whether or not to standardize predictors
    """
    results = {
        '{}_metrics'.format(exp_string): [],
        '{}_auc'.format(exp_string): [],
        '{}_aupr'.format(exp_string): [],
        '{}_coef'.format(exp_string): [],
    }
    signal = 'shuffled' if shuffle_labels else 'signal'

    for fold_no in range(num_folds):

        try:
            with warnings.catch_warnings():
                # sklearn warns us if one of the stratification classes has fewer
                # members than num_folds: in our case that will be the 'other'
                # class, and it's fine to distribute those unevenly. so here we
                # can ignore that warning.
                warnings.filterwarnings('ignore',
                                        message='The least populated class in y')
                X_train_raw_df, X_test_raw_df, _ = split_stratified(
                   data_model.X_df, sample_info, num_folds=num_folds,
                   fold_no=fold_no, seed=data_model.seed)
        except ValueError:
            if data_model.X_df.shape[0] == 0:
                raise NoTrainSamplesError(
                    'No train samples found for identifier: {}'.format(
                        identifier)
                )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df,
                                                   X_test_raw_df,
                                                   data_model.gene_features,
                                                   standardize_columns,
                                                   data_model.subset_mad_genes)

        try:
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
        except ValueError as e:
            if 'Only one class' in str(e):
                raise OneClassError(
                    'Only one class present in test set for identifier: '
                    '{}'.format(identifier)
                )
            else:
                # if not only one class error, just re-raise
                raise e

        (cv_pipeline,
         y_pred_train_df,
         y_pred_test_df,
         y_cv_df) = model_results

        # get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=X_train_df.columns,
            signal=signal,
            seed=data_model.seed
        )
        coef_df = coef_df.assign(identifier=identifier)
        coef_df = coef_df.assign(training_data=training_data)
        coef_df = coef_df.assign(fold=fold_no)

        try:
            metric_df, auc_df, aupr_df = get_metrics(
                y_train_df, y_test_df, y_cv_df, y_pred_train_df,
                y_pred_test_df, identifier, training_data, signal,
                data_model.seed, fold_no
            )
        except ValueError as e:
            if 'Only one class' in str(e):
                raise OneClassError(
                    'Only one class present in test set for identifier: '
                    '{}'.format(identifier)
                )
            else:
                # if not only one class error, just re-raise
                raise e

        results['{}_metrics'.format(exp_string)].append(metric_df)
        results['{}_auc'.format(exp_string)].append(auc_df)
        results['{}_aupr'.format(exp_string)].append(aupr_df)
        results['{}_coef'.format(exp_string)].append(coef_df)

    return results


def get_threshold_metrics(y_true, y_pred, drop=False):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments
    ---------
    y_true: an array of gold standard mutation status
    y_pred: an array of predicted mutation status
    drop: boolean if intermediate thresholds are dropped

    Returns
    -------
    dict of AUROC, AUPR, pandas dataframes of ROC and PR data, and cancer-type
    """
    roc_columns = ["fpr", "tpr", "threshold"]
    pr_columns = ["precision", "recall", "threshold"]

    roc_results = roc_curve(y_true, y_pred, drop_intermediate=drop)
    roc_items = zip(roc_columns, roc_results)
    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average="weighted")
    aupr = average_precision_score(y_true, y_pred, average="weighted")

    return {"auroc": auroc, "aupr": aupr, "roc_df": roc_df, "pr_df": pr_df}


def get_metrics(y_train_df, y_test_df, y_cv_df, y_pred_train, y_pred_test,
                identifier, training_data, signal, seed, fold_no):

    # get classification metric values
    y_train_results = get_threshold_metrics(
        y_train_df.status, y_pred_train, drop=False
    )
    y_test_results = get_threshold_metrics(
        y_test_df.status, y_pred_test, drop=False
    )
    y_cv_results = get_threshold_metrics(
        y_train_df.status, y_cv_df, drop=False
    )

    # summarize all results in dataframes
    metric_cols = [
        "auroc",
        "aupr",
        "identifier",
        "training_data",
        "signal",
        "seed",
        "data_type",
        "fold"
    ]
    train_metrics_, train_roc_df, train_pr_df = summarize_results(
        y_train_results, identifier, training_data, signal,
        seed, "train", fold_no
    )
    test_metrics_, test_roc_df, test_pr_df = summarize_results(
        y_test_results, identifier, training_data, signal,
        seed, "test", fold_no
    )
    cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
        y_cv_results, identifier, training_data, signal,
        seed, "cv", fold_no
    )

    # compile summary metrics
    metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
    metric_df = pd.DataFrame(metrics_, columns=metric_cols)
    auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
    aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])

    return metric_df, auc_df, aupr_df


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


def extract_coefficients(cv_pipeline, feature_names, signal, seed):
    """
    Pull out the coefficients from the trained classifiers

    Arguments
    ---------
    cv_pipeline: the trained sklearn cross validation pipeline
    feature_names: the column names of the x matrix used to train model (features)
    results: a results object output from `get_threshold_metrics`
    signal: the signal of interest
    seed: the seed used to compress the data
    """
    final_pipeline = cv_pipeline.best_estimator_
    final_classifier = final_pipeline.named_steps["classify"]

    coef_df = pd.DataFrame.from_dict(
        {"feature": feature_names, "weight": final_classifier.coef_[0]}
    )

    coef_df = (
        coef_df.assign(abs=coef_df["weight"].abs())
        .sort_values("abs", ascending=False)
        .reset_index(drop=True)
        .assign(signal=signal, seed=seed)
    )

    return coef_df


def summarize_results(results,
                      identifier,
                      training_data,
                      signal,
                      seed,
                      data_type,
                      fold_no):
    """
    Given an input results file, summarize and output all pertinent files

    Arguments
    ---------
    results: a results object output from `get_threshold_metrics`
    identifier: string describing the label being predicted
    training_data: the data type being used to train the model
    signal: the signal of interest
    seed: the seed used to compress the data
    data_type: the type of data (either training, testing, or cv)
    fold_no: the fold number for the external cross-validation loop
    """
    results_append_list = [
        identifier,
        training_data,
        signal,
        seed,
        data_type,
        fold_no,
    ]

    metrics_out_ = [results["auroc"], results["aupr"]] + results_append_list

    roc_df_ = results["roc_df"]
    pr_df_ = results["pr_df"]

    roc_df_ = roc_df_.assign(
        predictor=identifier,
        training_data=training_data,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    pr_df_ = pr_df_.assign(
        predictor=identifier,
        training_data=training_data,
        signal=signal,
        seed=seed,
        data_type=data_type,
        fold_no=fold_no,
    )

    return metrics_out_, roc_df_, pr_df_


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

