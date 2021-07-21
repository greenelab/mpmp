from pathlib import Path

import pandas as pd

def load_survival_results(results_dir):
    """Load results of survival prediction experiments.

    Arguments
    ---------
    results_dir (str): directory to look in for results, subdirectories should
                       be experiments for individual genes or cancer types

    Returns
    -------
    results_df (pd.DataFrame): summarizes experiment results
    """
    results_df = pd.DataFrame()
    results_dir = Path(results_dir)
    for results_file in results_dir.iterdir():
        if not results_file.is_file(): continue
        results_filename = str(results_file.stem)
        if ('survival' not in results_filename or
            'metrics' not in results_filename): continue
        if results_filename[0] == '.': continue
        id_results_df = pd.read_csv(results_file, sep='\t')
        results_df = pd.concat((results_df, id_results_df))
    return results_df
