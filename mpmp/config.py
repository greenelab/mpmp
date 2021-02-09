import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# important subdirectories
data_dir = repo_root / 'data'
results_dir = repo_root / 'results'

# locations of saved data files
raw_data_dir = data_dir / 'raw'
pancan_data = data_dir / 'pancancer_data.pkl'
sample_counts = data_dir / 'tcga_sample_counts.tsv'

# location of sample info
sample_info_dir = data_dir / 'sample_info'
sample_info = data_dir / 'tcga_sample_identifiers.tsv'

# locations of processed multimodal data files
rnaseq_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
# use this as default methylation data for now
# this is 27K methylation data, 450K is too large to use unprocessed
methylation_data = data_dir / 'methylation_preprocessed' / 'methylation_processed_n10_i5.tsv.gz'
data_types = {
    'expression': rnaseq_data,
    'methylation': methylation_data,
}
# If true, use only the samples present in both the 27k and 450k methylation datsets
use_only_cross_data_samples = True

# locations of compressed multimodal data files
exp_compressed_dir = data_dir / 'exp_compressed'
me_compressed_dir = data_dir / 'me_compressed'
compressed_data_types = {
    'expression': exp_compressed_dir / 'exp_std_pc{}.tsv.gz',
    'me_27k': me_compressed_dir / 'me_27k_f10_i5_pc{}.tsv.gz',
    'me_450k': me_compressed_dir / 'me_450k_f10_i5_pc{}.tsv.gz',
}

# locations of subsampled data, for debugging and testing
subsampled_data_dir = data_dir / 'subsampled'
subsampled_expression = subsampled_data_dir / 'expression_subsampled.tsv.gz'
subsampled_methylation = subsampled_data_dir / 'methylation_subsampled.tsv.gz'
subsampled_data_types = {
    'expression': subsampled_expression,
    'methylation': subsampled_methylation,
}

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
subsample_to_smallest = False
