# Batch effect correction experiments

As a way to understand how much non-linear signal exists in our data, we wanted to try applying a linear correction to our dataset, for the mutation labels that we're trying to predict (essentially "regressing out" the linear signal). To do this, we used [the limma::removeBatchEffect method](https://rdrr.io/bioc/limma/src/R/removeBatchEffect.R).

Results can be seen in the `batch_correction.ipynb` and `bc_titration.ipynb` notebooks. To summarize, we found that the linear models perform considerably worse after batch effect correction for the predictive labels (showing that linear signal is being removed from the dataset). However, the non-linear (LightGBM) models perform near-perfectly, even when only a few features are corrected. This suggests that the BE correction process could be leaking information about the labels into the dataset, in a way that only the non-linear model is able to detect it.
