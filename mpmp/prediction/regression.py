"""
Functions for training regression models on TCGA data.

"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
)

def train_regressor(X_train,
                    X_test,
                    y_train,
                    alphas,
                    l1_ratios,
                    seed,
                    n_folds=4,
                    max_iter=1000):
    """
    Build the logic and sklearn pipelines to predict real-valued y from dataset x

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed real-valued labels
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training,
    testing, and cross validation
    """
    # Setup the regression parameters
    import mpmp.config as cfg
    reg_parameters = {
        "regress__alpha": alphas,
        "regress__l1_ratio": l1_ratios,
    }


    # ElasticNet seems to be less sensitive to initialization/parameter choice
    # than SGDRegressor, but could scale poorly to really large datasets
    estimator = Pipeline(
        steps=[
            (
                "regress",
                ElasticNet(
                    random_state=seed,
                )
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=reg_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring="neg_mean_squared_error",
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
        method="predict",
    )

    # Get all performance results
    y_predict_train = cv_pipeline.predict(X_train)
    y_predict_test = cv_pipeline.predict(X_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def get_preds(X_test_df, y_test_df, cv_pipeline, fold_no):

    # get predictions
    y_preds_test = cv_pipeline.predict(X_test_df)

    return pd.DataFrame({
        'fold_no': fold_no,
        'true_label': y_test_df.status,
        'predicted_output': y_preds_test
    }, index=y_test_df.index)


def get_metrics(y_train_df,
                y_test_df,
                y_cv_df,
                y_pred_train,
                y_pred_test,
                **kwargs):
    """Get regression metric values for given model predictions."""

    train_metrics = get_continuous_metrics(y_train_df.status, y_pred_train)
    cv_metrics = get_continuous_metrics(y_train_df.status, y_cv_df)
    test_metrics = get_continuous_metrics(y_test_df.status, y_pred_test)

    columns = list(train_metrics.keys()) + ['data_type'] + list(kwargs.keys())
    train_metrics = list(train_metrics.values()) + ['train'] + list(kwargs.values())
    cv_metrics = list(cv_metrics.values()) + ['cv'] + list(kwargs.values())
    test_metrics = list(test_metrics.values()) + ['test'] + list(kwargs.values())

    return pd.DataFrame([train_metrics, cv_metrics, test_metrics],
                        columns=columns)


def get_continuous_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2}


