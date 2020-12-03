import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent

# important subdirectories
data_dir = repo_root / 'data'
results_dir = repo_root / 'results'

# location of saved expression data
pancan_data = data_dir / 'pancancer_data.pkl'
rnaseq_data = data_dir / 'tcga_expression_matrix_processed.tsv.gz'
sample_counts = data_dir / 'tcga_sample_counts.tsv'
sample_info = data_dir / 'tcga_sample_identifiers.tsv'
