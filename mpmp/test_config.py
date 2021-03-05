"""
Configuration information for tests

"""
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# location of data relevant to running tests
test_data_dir = repo_root / 'tests' / 'data'
test_pancan_data = test_data_dir / 'pancancer_data_subsampled.pkl'
test_stratified_results = str(test_data_dir / 'stratified_results_{}.tsv')

# gene mutation info used in tests
test_genes = ['TP53', 'KRAS', 'ARID1A']

# gene/training data/classification combos for stratified CV model tests
stratified_gene_info = [('TP53', 'TSG'),
                        ('KRAS', 'Oncogene'),
                        ('ARID1A', 'TSG')]
