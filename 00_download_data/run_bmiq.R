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
sample_names <- rownames(beta)
beta_t <- as.data.frame(t(beta))
colnames(beta_t) <- sample_names

# run PBC method on all samples in parallel
# TODO: remove or skip invalid cols
# beta_skip <- beta_t[,c(1:1024,1026:9749,9751:11975)]
# beta_pbc <- champ.norm(beta=beta_skip, method='PBC', cores=4)
# write.table(beta_pbc, 'data/methylation_27k_pbc_normalized.tsv', sep='\t')

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

beta_bmiq <- apply(beta_t, 2, BMIQ_skip_errors)
write.table(beta_bmiq, 'data/methylation_27k_bmiq_normalized.tsv', sep='\t')

