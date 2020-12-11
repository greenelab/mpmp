"""
Functions for writing and processing output files

"""
from pathlib import Path

from mpmp.exceptions import ResultsFileExistsError

def make_cancer_type_dir(results_dir, cancer_type):
    """Create a directory for the given cancer type."""
    dirname = 'cancer_type'
    cancer_type_dir = Path(results_dir, dirname, cancer_type).resolve()
    cancer_type_dir.mkdir(parents=True, exist_ok=True)
    return cancer_type_dir

def check_cancer_type_file(cancer_type_dir,
                           cancer_type,
                           shuffle_labels):
    signal = 'shuffled' if shuffle_labels else 'signal'
    check_file = Path(cancer_type_dir,
                      "{}_{}_coefficients.tsv.gz".format(
                          cancer_type, signal)).resolve()
    if check_file.is_file():
        raise ResultsFileExistsError(
            'Results file already exists for cancer type: {}\n'.format(
                cancer_type)
        )
    return check_file


