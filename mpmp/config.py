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

# hyperparameters for classification experiments
folds = 3
max_iter = 200
alphas = [0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
l1_ratios = [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]

# repo/commit information to retrieve precomputed cancer gene information
# this is used in data_utilities.py
top50_base_url = "https://github.com/greenelab/BioBombe/raw"
top50_commit = "aedc9dfd0503edfc5f25611f5eb112675b99edc9"
vogelstein_base_url = "https://github.com/greenelab/pancancer/raw"
vogelstein_commit = "2a0683b68017fb226f4053e63415e4356191734f"

# data types to standardize columns for
standardize_data_types = ['expression']

# subsample data to smallest cancer type
# hopefully this will improve prediction for imbalanced cancer types
subsample_to_smallest = True

