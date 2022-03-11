"""
Functions for training classifiers on TCGA data.

Some of these functions are adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/scripts/tcga_util.py
"""
import numpy as np
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
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
)

import mpmp.config as cfg

def train_linear_classifier(X_train,
                            X_test,
                            y_train,
                            alphas,
                            l1_ratios,
                            seed,
                            n_folds=4,
                            max_iter=1000):
    """
    Train a linear (elastic net logistic regression) classifier.

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix (output from align_matrices())
    alphas: list of alphas to perform cross validation over
    l1_ratios: list of l1 mixing parameters to perform cross validation over
    seed: seed for random number generator
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """
    # Setup the classifier parameters
    clf_parameters = {
        'classify__alpha': alphas,
        'classify__l1_ratio': l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                'classify',
                SGDClassifier(
                    random_state=seed,
                    class_weight='balanced',
                    loss='log',
                    penalty='elasticnet',
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
        scoring='average_precision',
        return_train_score=True,
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train.status)

    # obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=X_train,
        y=y_train.status,
        cv=n_folds,
        method='decision_function',
    )

    # get all performance results
    y_predict_train = cv_pipeline.decision_function(X_train)
    y_predict_test = cv_pipeline.decision_function(X_test)

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def train_gb_classifier(X_train,
                        X_test,
                        y_train,
                        learning_rates,
                        alphas,
                        lambdas,
                        seed,
                        n_folds=4,
                        max_iter=1000):
    """
    Fit gradient-boosted tree classifier to training data, and generate predictions
    for test data.

    Arguments
    ---------
    X_train: pandas DataFrame of feature matrix for training data
    X_test: pandas DataFrame of feature matrix for testing data
    y_train: pandas DataFrame of processed y matrix (output from align_matrices())
    n_folds: int of how many folds of cross validation to perform
    max_iter: the maximum number of iterations to test until convergence

    Returns
    ------
    The full pipeline sklearn object and y matrix predictions for training, testing,
    and cross validation
    """
    from lightgbm import LGBMClassifier

    clf_parameters = {
        'classify__learning_rate': learning_rates,
        'classify__reg_alpha': alphas,
        'classify__reg_lambda': lambdas,
     }

    estimator = Pipeline(
        steps=[
            (
                'classify',
                LGBMClassifier(
                    random_state=seed,
                    class_weight='balanced',
                    max_depth=5,
                    n_estimators=100,
                    colsample_bytree=0.35
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring='average_precision',
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
        method='predict_proba',
    )[:, 1]

    # Get all performance results
    y_predict_train = cv_pipeline.predict_proba(X_train)[:, 1]
    y_predict_test = cv_pipeline.predict_proba(X_test)[:, 1]

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def train_mlp_classifier(X_train,
                         X_test,
                         y_train,
                         params,
                         seed,
                         n_folds=4,
                         max_iter=1000):
    """Train multi-layer perceptron classifier."""
    import torch.optim
    from torch.utils.data import Dataset
    from skorch import NeuralNetBinaryClassifier
    from skorch.helper import SliceDataset

    from mpmp.prediction.nn_models import LogisticRegression, ThreeLayerNet

    # TODO model toggle
    # model = LogisticRegression(input_size=X_train.shape[1])
    model = ThreeLayerNet(input_size=X_train.shape[1])

    clf_parameters = {
        'lr': params['learning_rate'],
        'module__dropout': params['dropout'],
        'optimizer__weight_decay': params['weight_decay'],
     }

    net = NeuralNetBinaryClassifier(
        model,
        max_epochs=max_iter,
        batch_size=256,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        verbose=0, # by default this prints loss for each epoch
        train_split=False,
        device='cuda'
    )

    cv_pipeline = RandomizedSearchCV(
        estimator=net,
        param_distributions=clf_parameters,
        n_iter=30,
        cv=n_folds,
        scoring='average_precision',
        return_train_score=True,
        verbose=2,
    )

    # convert dataframe to something that can be indexed by batch
    class MyDataset(Dataset):

        def __init__(self, df):
            import torch
            self.df = df
            self.X = torch.tensor(df.values, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], None

    dataset = MyDataset(X_train)
    Xs = SliceDataset(dataset)

    # labels have to be cast to floats
    # https://discuss.pytorch.org/t/multi-label-binary-classification-result-type-float-cant-be-cast-to-the-desired-output-type-long/117915/3
    cv_pipeline.fit(X=Xs, y=y_train.status.values.astype(np.float32))

    # obtain cross validation results
    # TODO: this isn't working
    # TODO: are these even useful in general? seems like it would be
    #       more useful to see results for each parameter choice
    # y_cv = cross_val_predict(
    #     cv_pipeline.best_estimator_,
    #     X=X_train.values.astype(np.float32),
    #     y=y_train.status.values,
    #     cv=n_folds,
    #     method='predict_proba',
    # )[:, 1]

    # get all performance results
    y_predict_train = cv_pipeline.predict_proba(X_train.values.astype(np.float32))[:, 1]
    y_predict_test = cv_pipeline.predict_proba(X_test.values.astype(np.float32))[:, 1]
    y_cv = y_predict_train

    return cv_pipeline, y_predict_train, y_predict_test, y_cv


def get_preds(X_test_df, y_test_df, cv_pipeline, fold_no):
    """Get model-predicted probability of positive class for test data.

    Also returns true class, to enable quantitative comparisons in analyses.
    """

    # get probability of belonging to positive class
    y_scores_test = cv_pipeline.decision_function(X_test_df)
    y_probs_test = cv_pipeline.predict_proba(X_test_df)

    # make sure we're actually looking at positive class prob
    assert np.array_equal(cv_pipeline.best_estimator_.classes_,
                          np.array([0, 1]))

    return pd.DataFrame({
        'fold_no': fold_no,
        'true_class': y_test_df.status,
        'score': y_scores_test,
        'positive_prob': y_probs_test[:, 1]
    }, index=y_test_df.index)


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
    if not isinstance(training_data, str):
        training_data = '.'.join(training_data)

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


