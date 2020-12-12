"""
Exceptions specific to pan-cancer prediction experiments
"""

class ResultsFileExistsError(Exception):
    """
    Custom exception to raise when the results file already exists for the
    given gene and cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)


class NoTrainSamplesError(Exception):
    """
    Custom exception to raise when there are no train samples in a
    cross-validation fold for a given cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)


class NoTestSamplesError(Exception):
    """
    Custom exception to raise when there are no test samples in a
    cross-validation fold for a given cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)

