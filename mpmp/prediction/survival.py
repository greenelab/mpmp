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
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """

    # TODO: does this help?
    X_train.age = (X_train.age - X_train.age.mean()) / X_train.age.std()
    X_test.age = (X_test.age - X_test.age.mean()) / X_test.age.std()

    # set up the cross-validation parameters
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
                    tol=1e-3,
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


    if debug_info is not None:
        # TODO: does this need to be done separately from standard grid?
        grid_results = []
        for i, alpha in enumerate(alphas):
            grid_results.append([])
            for l1_ratio in l1_ratios:
                fit = 'fit'
                print('alpha: {}, l1_ratio: {}'.format(alpha, l1_ratio))
                # catch convergence warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    try:
                        cox = CoxnetSurvivalAnalysis(max_iter=max_iter,
                                                     tol=1e-6,
                                                     alphas=[alpha],
                                                     l1_ratio=l1_ratio)
                        cox.fit(X_train, _y_df_to_struct(y_train))
                    except UserWarning:
                        fit = 'too small'
                        print('fit failed, all coefficients are zero')
                    except ArithmeticError:
                        fit = 'too large'
                        print('fit failed, coefficients are too large')
                    except ConvergenceWarning:
                        print('convergence warning')
                    grid_results[i].append(fit)

        grid_results_df = pd.DataFrame(grid_results,
                                       index=alphas,
                                       columns=l1_ratios)
        grid_results_df.index.name = 'alphas'
        grid_results_df.columns.name = 'alphas'
        grid_results_df.to_csv('{}_{}_fold{}_grid.tsv'.format(debug_info['prefix'],
                                                              debug_info['signal'],
                                                              debug_info['fold_no']),
                                sep='\t')

    # fit the model
    # TODO: catch warnings?
    try:
        cv_pipeline.fit(X=X_train,
                        y=_y_df_to_struct(y_train))
    except ValueError:
        # this happens when all samples are censored
        raise OneClassError

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


def _y_df_to_struct(y_df):
    return np.core.records.fromarrays(
               y_df.loc[:, ['status', 'time_in_days']].values.T,
               names='status, time_in_days',
               formats='?, <f8'
           )
