# use implementation of PBC probe bias correction in ChAMP package
suppressMessages(library(ChAMP))
# use implementation of BMIQ probe bias correction in wateRmelon package
suppressMessages(library(wateRmelon))

set.seed(42)

# load filtered probe data generated in methylation_beta.ipynb
beta <- read.table('data/methylation_27k_filtered_probes.tsv',
                   header=T, sep='\t', row.names=1)
probe_types <- read.table('data/methylation_27k_filtered_probe_types.txt',
                          header=T, sep=',', row.names=1)

# transpose beta so probes are rows, samples are columns
# https://stackoverflow.com/a/6779744
transpose_df <- function(df) {
    sample_names <- rownames(df)
    df_t <- as.data.frame(t(df))
    colnames(df_t) <- sample_names
    df_t
}
beta_t <- transpose_df(beta)

# run BMIQ method on each sample (column)
# BMIQ throws errors if beta mixture classes are too imbalanced,
# if this is the case we'll just skip that probe
counter <- 0
BMIQ_skip_errors <- function(beta) {
    tryCatch({
        beta_norm <- BMIQ(beta, design.v=probe_types[,1], nfit=1000, plots=F, pri=F)
        ret <- beta_norm$nbeta
    }, error = function(e) {ret <<- NA});
    counter <<- counter + 1
    print(paste('iter', counter, 'of', dim(beta_t)[2]))
    ret
}
beta_t_bmiq <- apply(beta_t, 2, BMIQ_skip_errors)

# this writes the results as a probes X samples tsv file
# also note that R replaces dashes/hyphens in column names with dots,
# so these need to be converted back in the script that loads this data
# (currently the methylation_beta.ipynb notebook)
write.table(beta_t_bmiq, 'data/methylation_27k_bmiq_normalized.tsv', sep='\t')

