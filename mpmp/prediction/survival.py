"""
Functions for training survival prediction models on TCGA data.

"""
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
)
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis

import mpmp.config as cfg
from mpmp.exceptions import OneClassError

def train_survival(X_train,
                   X_test,
                   y_train,
                   alphas,
                   l1_ratios,
                   seed,
                   n_folds=4,
                   max_iter=1000,
                   output_fn=False,
                   debug_info=None):
    """
    Build the logic and sklearn pipelines to predict survival info y from dataset x,
    using elastic net Cox regression

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix, containing 'status' = False
             if right-censored else True, 'time_in_days' = survival time
    alphas: list of alphas to perform cross validation over, if None use the
            alphas path generated by scikit-survival
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """

    # set up the cross-validation parameters
    # sometimes we want to use sksurv to compute the alpha path
    if alphas is None:
        cox = CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, n_alphas=100)
        cox.fit(X_train, _y_df_to_struct(y_train))
        alphas = cox.alphas_

    surv_parameters = {
        "survival__alphas": [[a] for a in alphas],
        "survival__l1_ratio": l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                "survival",
                CoxnetSurvivalAnalysis(
                    max_iter=max_iter,
                    tol=1e-5,
                    # normalize input features
                    # this seems to help with model convergence
                    # TODO could try doing this using StandardScaler pipeline
                    normalize=True,
                    fit_baseline_model=output_fn
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=surv_parameters,
        n_jobs=-1,
        cv=n_folds,
        error_score=0.5,
        return_train_score=True,
    )

    # fit the model
    cv_pipeline.fit(X=X_train,
                    y=_y_df_to_struct(y_train))

    if debug_info is not None:
        grid_mean_df = pd.DataFrame(
            cv_pipeline.cv_results_['mean_test_score'].reshape(len(alphas), -1),
            columns=l1_ratios,
            index=alphas
        )
        grid_mean_df.to_csv('{}_{}_fold{}_grid.tsv'.format(debug_info['prefix'],
                                                           debug_info['signal'],
                                                           debug_info['fold_no']),
                            sep='\t')

    # Obtain cross validation results
    y_cv = cross_val_predict(
       cv_pipeline.best_estimator_,
       X=X_train,
       y=_y_df_to_struct(y_train),
       cv=n_folds,
       method="predict",
    )

    # get predictions
    y_predict_train = cv_pipeline.predict(X_train)
    y_predict_test = cv_pipeline.predict(X_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def get_metrics(cv_pipeline,
                X_train_df,
                X_test_df,
                X_cv_df,
                y_train_df,
                y_test_df,
                y_cv_df,
                **kwargs):
    """Get survival metric values for fit model/CV pipeline."""

    train_metrics = get_survival_metrics(cv_pipeline, X_train_df, y_train_df)
    cv_metrics = get_survival_metrics(cv_pipeline, X_cv_df, y_train_df)
    test_metrics = get_survival_metrics(cv_pipeline, X_test_df, y_test_df)

    columns = list(train_metrics.keys()) + ['data_type'] + list(kwargs.keys())
    train_metrics = list(train_metrics.values()) + ['train'] + list(kwargs.values())
    cv_metrics = list(cv_metrics.values()) + ['cv'] + list(kwargs.values())
    test_metrics = list(test_metrics.values()) + ['test'] + list(kwargs.values())

    return pd.DataFrame([train_metrics, cv_metrics, test_metrics],
                        columns=columns)


def get_survival_metrics(cv_pipeline, X_df, y_df):
    try:
        cindex = cv_pipeline.score(X_df, _y_df_to_struct(y_df))
    except ValueError: # all samples are censored
        cindex = 0.0
    # TODO add more?
    return {'cindex': cindex}


def get_survival_function(cv_pipeline, X_test_df):
    """Get model-predicted survival function for test data."""
    return {
        'samples': X_test_df.index.values,
        'functions': cv_pipeline.best_estimator_.predict_survival_function(X_test_df)
    }


def _y_df_to_struct(y_df):
    return np.core.records.fromarrays(
               y_df.loc[:, ['status', 'time_in_days']].values.T,
               names='status, time_in_days',
               formats='?, <f8'
           )
