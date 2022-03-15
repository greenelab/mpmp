"""
Check if expected output files are present.

Runs for the current working directory by default. Prints missing filenames
to stdout, separated by newlines.
"""
import itertools as it
from pathlib import Path

expected_params = {
    'gene': ['TP53'],
    'data_types': ['expression', 'me_27k', 'me_450k'],
    'signal': ['signal', 'shuffled'],
    'seed': [42, 1],
    'n_dim': [5000],
    'fold': [0, 1, 2, 3],
}

results_file_types = [
    'metrics',
    'coefficients',
    'param_grid'
]

if __name__ == '__main__':
    missing_files = []
    for file_id in it.product(*expected_params.values()):
        fname_stem = Path.cwd() / (
            '{}_{}_{}_classify_s{}_n{}_f{}_{{}}.tsv.gz'.format(*file_id)
        )
        for file_type in results_file_types:
            fname = Path(str(fname_stem).format(file_type))
            if not fname.is_file():
                missing_files.append(fname)

    print('\n'.join([str(f) for f in missing_files]))

