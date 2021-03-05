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
expression_sample_info = sample_info_dir / 'tcga_expression_sample_identifiers.tsv'
me_27k_sample_info = sample_info_dir / 'tcga_me_27k_sample_identifiers.tsv'
me_450k_sample_info = sample_info_dir / 'tcga_me_450k_sample_identifiers.tsv'
rppa_sample_info = sample_info_dir / 'tcga_rppa_sample_identifiers.tsv'
mut_sigs_sample_info = sample_info_dir / 'tcga_mut_sigs_sample_identifiers.tsv'
sample_infos = {
    'expression': expression_sample_info,
    'me_27k': me_27k_sample_info,
    'me_450k': me_450k_sample_info,
    'rppa': rppa_sample_info,
    'mut_sigs': mut_sigs_sample_info,
}

# locations of processed multimodal data files
expression_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
methylation_27k_data = data_dir / 'me_preprocessed' / 'methylation_processed_n10_i5.tsv.gz'
methylation_450k_data = data_dir / 'methylation_450k_f10_i5_mad100000.pkl'
rppa_data = data_dir / 'tcga_rppa_matrix_processed.tsv'
mut_sigs_data = data_dir / 'tcga_wes_sbs_mutational_signatures.tsv'
data_types = {
    'expression': expression_data,
    'me_27k': methylation_27k_data,
    'me_450k': methylation_450k_data,
    'rppa': rppa_data,
    'mut_sigs': mut_sigs_data,
}
# if true, use only the samples present in all datasets
# if false, use all the samples present in the dataset being analyzed
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

# repo/commit information to retrieve TCGA code -> (sample|cancer) type map
# we get this from cognoma: https://github.com/cognoma/cancer-data/
sample_commit = 'da832c5edc1ca4d3f665b038d15b19fced724f4c'
cancer_types_url = (
    'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_cancertype_codes.csv'.format(
        sample_commit)
)
sample_types_url = (
    'https://raw.githubusercontent.com/cognoma/cancer-data/{}/mapping/tcga_sampletype_codes.csv'.format(
        sample_commit)
)

# data types to standardize columns for
standardize_data_types = ['expression', 'rppa']

# subsample data to smallest cancer type
# hopefully this will improve prediction for imbalanced cancer types
subsample_to_smallest = False
