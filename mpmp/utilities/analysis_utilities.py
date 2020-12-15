from pathlib import Path

import pandas as pd

def load_cancer_type_prediction_results(results_dir, experiment_descriptor):
    """Load results of cancer type prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual cancer types
    experiment_descriptor (str): string describing this experiment, can be
                                 useful to segment analyses involving multiple
                                 experiments or results sets

    Returns
    -------
    results_df (pd.DataFrame): results of classification experiments
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for cancer_type in results_dir.iterdir():
        cancer_type_dir = Path(results_dir, cancer_type)
        if cancer_type_dir.is_file(): continue
        for results_file in cancer_type_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.stem)
            if 'classify' not in results_filename: continue
            if results_filename[0] == '.': continue
            gene_results_df = pd.read_csv(results_file, sep='\t')
            gene_results_df['experiment'] = experiment_descriptor
            results_df = pd.concat((results_df, gene_results_df))
    return results_df

