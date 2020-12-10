import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# important subdirectories
data_dir = repo_root / 'data'
results_dir = repo_root / 'results'

# locations of saved data
raw_data_dir = data_dir / 'raw'
pancan_data = data_dir / 'pancancer_data.pkl'
rnaseq_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
methylation_data = data_dir / 'tcga_methylation_matrix_processed.tsv.gz'
sample_counts = data_dir / 'tcga_sample_counts.tsv'
sample_info = data_dir / 'tcga_sample_identifiers.tsv'

# locations of subsampled data, for debugging and testing
subsampled_data_dir = data_dir / 'subsampled'
subsampled_expression = subsampled_data_dir / 'expression_subsampled.tsv.gz'
subsampled_methylation = subsampled_data_dir / 'methylation_subsampled.tsv.gz'

# default seed for random number generator
default_seed = 42

# number of features to use by default
num_features_raw = 8000

# gene/cancer type filtering hyperparameters
# filter cancer types with less than this percent of mutated samples
filter_prop = 0.05
# filter cancer types with less than this number of mutated samples
filter_count = 15
