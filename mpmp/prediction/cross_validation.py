"""
Functions to split data for cross-validation.

"""
import warnings
import contextlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import mpmp.config as cfg
from mpmp.exceptions import (
    NoTrainSamplesError,
    NoTestSamplesError,
    OneClassError,
    ModelFitError,
)
import mpmp.prediction.classification as clf
import mpmp.prediction.regression as reg
import mpmp.prediction.survival as surv
import mpmp.utilities.tcga_utilities as tu

def run_cv_stratified(data_model,
                      exp_string,
                      identifier,
                      training_data,
                      sample_info,
                      num_folds,
                      predictor='classify',
                      shuffle_labels=False,
                      standardize_columns=False,
                      output_preds=False,
                      output_survival_fn=False,
                      survival_fit_ridge=False,
                      stratify=True,
                      nonlinear=False,
                      results_dir=None):
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
    predictor (str): one of 'classify', 'regress', 'survival'
    shuffle_labels (bool): whether or not to shuffle labels (negative control)
    standardize_columns (bool): whether or not to standardize predictors
    output_preds (bool): whether or not to write predictions to file

    Returns
    -------
    results (dict): maps results metrics to values across CV folds
    """
    if predictor == 'classify':
        results = {
            '{}_metrics'.format(exp_string): [],
            '{}_auc'.format(exp_string): [],
            '{}_aupr'.format(exp_string): [],
        }
    else:
        results = {
            '{}_metrics'.format(exp_string): [],
            '{}_coef'.format(exp_string): [],
        }
    signal = 'shuffled' if shuffle_labels else 'signal'

    if output_preds:
        results['{}_preds'.format(exp_string)] = []

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
                   fold_no=fold_no, seed=data_model.seed, stratify=stratify)
        except ValueError:
            if data_model.X_df.shape[0] == 0:
                raise NoTrainSamplesError(
                    'No train samples found for identifier: {}'.format(
                        identifier)
                )

        y_train_df = data_model.y_df.reindex(X_train_raw_df.index)
        y_test_df = data_model.y_df.reindex(X_test_raw_df.index)

        # shuffle labels for train/test sets separately
        # this ensures that overall label balance isn't affected
        # (see https://github.com/greenelab/mpmp/issues/44)
        if shuffle_labels:
            if cfg.shuffle_by_cancer_type:
                # in this case we want to shuffle labels independently for each cancer type
                # (i.e. preserve the total number of mutated samples in each)
                original_ones = y_train_df.groupby('DISEASE').sum()['status']
                y_train_df.status = shuffle_by_cancer_type(y_train_df, data_model.seed)
                y_test_df.status = shuffle_by_cancer_type(y_test_df, data_model.seed)
                new_ones = y_train_df.groupby('DISEASE').sum()['status']

                # number of mutated samples per cancer type should be the same before
                # and after shuffling
                assert original_ones.equals(new_ones)

            else:
                # we set a temp seed here to make sure this shuffling order
                # is the same for each gene between data types, otherwise
                # it might be slightly different depending on the global state
                with temp_seed(data_model.seed):
                    y_train_df.status = np.random.permutation(y_train_df.status.values)
                    y_test_df.status = np.random.permutation(y_test_df.status.values)

        # choose single-omics or multi-omics preprocessing function based on
        # data_model.gene_data_types class attribute
        if hasattr(data_model, 'gene_data_types'):
            X_train_df, X_test_df = tu.preprocess_multi_data(X_train_raw_df,
                                                             X_test_raw_df,
                                                             data_model.gene_features,
                                                             data_model.gene_data_types,
                                                             standardize_columns,
                                                             data_model.subset_mad_genes)
        else:
            X_train_df, X_test_df = tu.preprocess_data(X_train_raw_df,
                                                       X_test_raw_df,
                                                       data_model.gene_features,
                                                       standardize_columns,
                                                       data_model.subset_mad_genes)

        models_list = {
            'classify': (clf.train_gb_classifier if nonlinear
                         else clf.train_classifier),
            'regress': reg.train_regressor,
            'survival': surv.train_survival
        }
        train_model = models_list[predictor]

        # save model results for survival prediction
        if predictor == 'survival':
            train_model = partial(train_model,
                                  output_fn=output_survival_fn,
                                  fit_ridge=survival_fit_ridge)
        if predictor == 'survival' and cfg.survival_debug:
            debug_info = {
                'fold_no': fold_no,
                'prefix': '{}/{}_{}'.format(
                    results_dir, identifier, predictor
                ),
                'signal': signal
            }
            # the non-survival model training functions don't take a debug_info
            # parameter, so we do a partial function application to make all the
            # model training functions take the same arguments
            train_model = partial(train_model, debug_info=debug_info)

        # set the hyperparameters
        train_model_params = apply_model_params(train_model, predictor, nonlinear)

        try:
            model_results = train_model_params(
                X_train=X_train_df,
                X_test=X_test_df,
                y_train=y_train_df,
                seed=data_model.seed,
                n_folds=cfg.folds,
                max_iter=cfg.max_iter_map[predictor],
            )
        except ValueError as e:
            if ('Only one class' in str(e)) or ('got 1 class' in str(e)):
                raise OneClassError(
                    'Only one class present in test set for identifier: '
                    '{}'.format(identifier)
                )
            elif ('All samples are censored' in str(e)):
                raise OneClassError(
                    'All samples are censored in test set for identifier:'
                    '{}'.format(identifier)
                )
            elif ('search direction contains NaN' in str(e)):
                raise ModelFitError(
                    'Hyperparameter path returned NaN/infinite results for '
                    'identifier: {}'.format(identifier)
                )
            else:
                # if not only one class error, just re-raise
                raise e

        (cv_pipeline,
         y_pred_train,
         y_pred_test,
         y_cv_df) = model_results

        # get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=X_train_df.columns,
            signal=signal,
            seed=data_model.seed,
            name=predictor,
            nonlinear=nonlinear
        )
        coef_df = coef_df.assign(identifier=identifier)
        if isinstance(training_data, str):
            coef_df = coef_df.assign(training_data=training_data)
        else:
            coef_df = coef_df.assign(training_data='.'.join(training_data))
        coef_df = coef_df.assign(fold=fold_no)
        if '{}_coef'.format(exp_string) not in results:
            results['{}_coef'.format(exp_string)] = [coef_df]
        else:
            results['{}_coef'.format(exp_string)].append(coef_df)

        # get relevant metrics
        if predictor == 'classify':
            try:
                metric_df, auc_df, aupr_df = clf.get_metrics(
                    y_train_df, y_test_df, y_cv_df, y_pred_train,
                    y_pred_test, identifier, training_data, signal,
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

        else:
            if predictor == 'survival':
                metric_df = surv.get_metrics(
                    cv_pipeline,
                    X_train_df,
                    X_test_df,
                    X_train_df,
                    y_train_df,
                    y_test_df,
                    y_cv_df,
                    identifier=identifier,
                    training_data=training_data,
                    signal=signal,
                    seed=data_model.seed,
                    fold_no=fold_no
                )
            else:
                metric_df = reg.get_metrics(
                    y_train_df,
                    y_test_df,
                    y_cv_df,
                    y_pred_train,
                    y_pred_test,
                    identifier=identifier,
                    training_data=training_data,
                    signal=signal,
                    seed=data_model.seed,
                    fold_no=fold_no
                )
            results['{}_metrics'.format(exp_string)].append(metric_df)

        if output_preds:
            if predictor == 'survival':
                raise NotImplementedError
            get_preds = clf.get_preds if predictor == 'classify' else reg.get_preds
            results['{}_preds'.format(exp_string)].append(
                get_preds(X_test_df, y_test_df, cv_pipeline, fold_no)
            )

        if output_survival_fn:
            import pickle as pkl
            if predictor != 'survival':
                raise NotImplementedError
            surv_fns = surv.get_survival_function(cv_pipeline, X_test_df)
            fn_prefix = '{}/{}_{}'.format(results_dir, identifier, predictor)
            fn_file = '{}_{}_fold{}_functions.pkl'.format(fn_prefix,
                                                          signal,
                                                          fold_no)
            with open(fn_file, 'wb') as f:
                pkl.dump(surv_fns, f)

    return results


def split_stratified(data_df,
                     sample_info_df,
                     num_folds=4,
                     fold_no=1,
                     seed=cfg.default_seed,
                     stratify=True):
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
    if stratify: # TODO this is a mess, clean up
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_ixs, test_ixs) in enumerate(
                kf.split(data_df, sample_info_df.id_for_stratification)):
            if fold == fold_no:
                train_df = data_df.iloc[train_ixs]
                test_df = data_df.iloc[test_ixs]
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_ixs, test_ixs) in enumerate(kf.split(data_df)):
            if fold == fold_no:
                train_df = data_df.iloc[train_ixs]
                test_df = data_df.iloc[test_ixs]

    return train_df, test_df, sample_info_df


def extract_coefficients(cv_pipeline,
                         feature_names,
                         signal,
                         seed,
                         name='classify',
                         nonlinear=False):
    """
    Pull out the coefficients from the trained models

    Arguments
    ---------
    cv_pipeline: the trained sklearn cross validation pipeline
    feature_names: the column names of the x matrix used to train model (features)
    results: a results object output from `get_threshold_metrics`
    signal: the signal of interest
    seed: the seed used to compress the data
    """
    final_pipeline = cv_pipeline.best_estimator_
    final_classifier = final_pipeline.named_steps[name]

    if name == 'survival':
        weights = final_classifier.coef_.flatten()
    elif nonlinear:
        # use information gain for feature importance with nonlinear classifier
        weights = final_classifier.booster_.feature_importance(importance_type='gain')
    else:
        weights = final_classifier.coef_[0]

    coef_df = pd.DataFrame.from_dict(
        {"feature": feature_names, "weight": weights}
    )

    coef_df = (coef_df
        .assign(abs=coef_df["weight"].abs())
        .sort_values("abs", ascending=False)
        .reset_index(drop=True)
        .assign(signal=signal, seed=seed)
    )

    return coef_df


def shuffle_by_cancer_type(y_df, seed):
    y_copy_df = y_df.copy()
    with temp_seed(seed):
        for cancer_type in y_copy_df.DISEASE.unique():
            is_cancer_type = (y_copy_df.DISEASE == cancer_type)
            y_copy_df.loc[is_cancer_type, 'status'] = (
                np.random.permutation(y_copy_df.loc[is_cancer_type, 'status'].values)
            )
    return y_copy_df.status.values


@contextlib.contextmanager
def temp_seed(cntxt_seed):
    """Set a temporary np.random seed in the resulting context.

    This saves the global random number state and puts it back once the context
    is closed. See https://stackoverflow.com/a/49557127 for more detail.
    """
    state = np.random.get_state()
    np.random.seed(cntxt_seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def apply_model_params(train_model, predictor, nonlinear):
    if predictor == 'classify' and nonlinear:
        # non-linear classifier takes different hyperparameters,
        # pass them through here
        return partial(
            train_model,
            learning_rates=cfg.learning_rates,
            alphas=cfg.alphas,
            lambdas=cfg.lambdas,
        )
    else:
        # all other models use alphas and l1_ratios
        # (i.e. elastic net hyperparameters)
        return partial(
            train_model,
            alphas=cfg.alphas_map[predictor],
            l1_ratios=cfg.l1_ratios_map[predictor],
        )


