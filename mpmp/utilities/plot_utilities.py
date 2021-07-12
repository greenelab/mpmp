import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Rectangle

import mpmp.utilities.analysis_utilities as au

def plot_volcano_baseline(results_df,
                          axarr,
                          training_data_map,
                          sig_alpha,
                          sig_alphas=[0.05, 0.01, 0.001],
                          metric='aupr',
                          xlim=None,
                          ylim=None,
                          verbose=False,
                          color_overlap=False):
    """Make a scatter plot comparing classifier results to shuffled baseline.

    Arguments
    ---------
    results_df (pd.DataFrame): dataframe with processed results/p-values
    axarr (array of Axes): axes array from plt.subplots
    training_data_map (dict): maps internal data type names to display names
    sig_alpha (float): significance testing threshold
    """

    # set axis limits if not provided
    if xlim is None:
        xlim = (-0.2, 1.0)
    if ylim is None:
        y_max = results_df.nlog10_p.max()
        ylim = (0, y_max+3)

    # plot mutation prediction from expression, in a volcano-like plot
    for ix, training_data in enumerate(training_data_map.values()):

        try:
            if axarr.ndim > 1:
                ax = axarr[ix // axarr.shape[1], ix % axarr.shape[1]]
            else:
                ax = axarr[ix]
        except AttributeError:
            # no axarr.ndim => only a single axis
            ax = axarr

        data_results_df = results_df[results_df.training_data == training_data]

        sns.scatterplot(data=data_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                        hue_order=[False, True], ax=ax, legend=(ix == 0))

        if color_overlap:
            overlap_genes = _get_overlap_genes(results_df,
                                               training_data)
            overlap_df = data_results_df[data_results_df.gene.isin(overlap_genes)]
            sns.scatterplot(data=overlap_df, x='delta_mean', y='nlog10_p',
                            color='red', ax=ax, legend=False)

        # add vertical line at 0
        ax.axvline(x=0, linestyle='--', linewidth=1.25, color='black')

        for alpha in sig_alphas:

            # add horizontal line at statistical significance threshold
            l = ax.axhline(y=-np.log10(alpha), linestyle='--', linewidth=1.25)

            # label horizontal line with significance threshold
            # (matplotlib makes this fairly difficult, sadly)
            ax.text(0.875, -np.log10(alpha)+0.02,
                    r'$\mathbf{{\alpha = {}}}$'.format(alpha),
                    va='center', ha='center', color=l.get_color(),
                    backgroundcolor=ax.get_facecolor())

        # label axes and set axis limits
        ax.set_xlabel('{}(signal) - {}(shuffled)'.format(
                          metric.upper(), metric.upper()),
                      size=14)
        ax.set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$', size=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # only add a legend to the first subplot
        if ix == 0:
            ax.legend(title=r'Reject $H_0$', loc='upper left',
                      fontsize=14, title_fontsize=14)
        ax.set_title(r'Mutation prediction, {} data'.format(training_data), size=14)

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand in inkscape if necessary
        text_labels = _label_points(data_results_df['delta_mean'],
                                    data_results_df['nlog10_p'],
                                    data_results_df.gene,
                                    ax,
                                    sig_alpha)
        adjust_text(text_labels,
                    ax=ax,
                    expand_text=(1., 1.),
                    lim=5)

        if verbose:
            # print significant gene count for each training data type
            print('{}: {}/{}'.format(
                training_data,
                np.count_nonzero(data_results_df.reject_null),
                data_results_df.shape[0]
            ))


def plot_volcano_comparison(results_df,
                            axarr,
                            training_data_map,
                            sig_alpha,
                            sig_alphas=[0.05, 0.01, 0.001],
                            metric='aupr',
                            xlim=None,
                            ylim=None,
                            verbose=False):
    """Make a scatter plot comparing classifier results to expression.

    Arguments
    ---------
    results_df (pd.DataFrame): dataframe with processed results/p-values
    axarr (array of Axes): axes array from plt.subplots
    training_data_map (dict): maps internal data type names to display names
    sig_alpha (float): significance testing threshold
    """

    # set axis limits if not provided
    if xlim is None:
        xlim = (-0.75, 0.75)
    if ylim is None:
        ylim = (0, 8)

    data_types = sorted([dt for dt in training_data_map.values()
                           if dt != 'gene expression'])
    exp_results_df = results_df[results_df.training_data == 'gene expression'].copy()
    exp_results_df.sort_values(by=['seed', 'fold'], inplace=True)

    for ix, training_data in enumerate(data_types):

        try:
            if axarr.ndim > 1:
                ax = axarr[ix // axarr.shape[1], ix % axarr.shape[1]]
            else:
                ax = axarr[ix]
        except AttributeError:
            # no axarr.ndim => only a single axis
            ax = axarr

        data_results_df = results_df[results_df.training_data == training_data].copy()
        data_results_df.sort_values(by=['seed', 'fold'], inplace=True)
        compare_results_df = au.compare_results(exp_results_df,
                                                condition_2_df=data_results_df,
                                                identifier='identifier',
                                                metric=metric,
                                                correction=True,
                                                correction_method='fdr_bh',
                                                correction_alpha=sig_alpha,
                                                verbose=True)
        compare_results_df.rename(columns={'identifier': 'gene'}, inplace=True)
        compare_results_df['nlog10_p'] = -np.log10(compare_results_df.corr_pval)

        sns.scatterplot(data=compare_results_df, x='delta_mean', y='nlog10_p',
                        hue='reject_null', hue_order=[False, True], ax=ax,
                        legend=(ix == 0))

        # add vertical line at 0
        ax.axvline(x=0, linestyle='--', linewidth=1.25, color='black')

        for alpha in sig_alphas:

            # add horizontal line at statistical significance threshold
            l = ax.axhline(y=-np.log10(alpha), linestyle='--', linewidth=1.25)

            # label horizontal line with significance threshold
            # (matplotlib makes this fairly difficult, sadly)
            ax.text(0.5, -np.log10(alpha)+0.01,
                    r'$\mathbf{{\alpha = {}}}$'.format(alpha),
                    va='center', ha='center', color=l.get_color(),
                    backgroundcolor=ax.get_facecolor())

        # label axes and set axis limits
        ax.set_xlabel('{}({}) - {}(expression)'.format(
                          metric.upper(), training_data, metric.upper()),
                      size=14)
        ax.set_ylabel(r'$-\log_{10}($adjusted $p$-value$)$', size=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # only add a legend to the first subplot
        if ix == 0:
            ax.legend(title=r'Reject $H_0$', loc='upper left',
                      fontsize=14, title_fontsize=14)

        ax.set_title(
            r'Mutation prediction, expression vs. {}'.format(training_data),
            size=14
        )

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand if necessary
        text_labels = _label_points_compare(
                          compare_results_df['delta_mean'],
                          compare_results_df['nlog10_p'],
                          compare_results_df.gene,
                          ax,
                          sig_alpha)
        adjust_text(text_labels,
                    ax=ax,
                    expand_text=(1., 1.),
                    lim=5)

        if verbose:
            print('{}: {}/{}'.format(training_data,
                                     np.count_nonzero(compare_results_df.reject_null),
                                     compare_results_df.shape[0]))


def plot_boxes(results_df,
               axarr,
               training_data_map,
               metric='aupr',
               orientation='h',
               verbose=False,
               pairwise_tests=False,
               pairwise_box_pairs=None,
               plot_significant=True):
    """Make a box plot comparing classifier results between data types.

    Arguments
    ---------
    results_df (pd.DataFrame): dataframe with processed results/p-values
    axarr (array of Axes): axes array from plt.subplots
    training_data_map (dict): maps internal data type names to display names
    """

    # plot mean performance over all genes in Vogelstein dataset
    try:
        ax_all = axarr[0]
    except TypeError:
        # no axarr.ndim => only a single axis
        ax_all = axarr
    sns.boxplot(data=results_df, x='training_data', y='delta_mean',
                ax=ax_all, order=list(training_data_map.values()))
    ax_all.set_title('Prediction for all genes, performance vs. data type', size=14)
    if orientation == 'v':
        ax_all.set_xlabel('')
    else:
        ax_all.set_xlabel('Data type', size=14)
    ax_all.set_ylabel('{}(signal) - {}(shuffled)'.format(
                      metric.upper(), metric.upper()),
                  size=14)
    ax_all.set_ylim(-0.2, 0.7)
    for tick in ax_all.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_rotation(30)

    if plot_significant:
        # plot mean performance for genes that are significant for at least one data type
        ax_sig = axarr[1]
        gene_list = results_df[results_df.reject_null == True].gene.unique()
        if verbose:
            print(gene_list.shape)
            print(gene_list)
        sns.boxplot(data=results_df[results_df.gene.isin(gene_list)],
                    x='training_data', y='delta_mean', ax=ax_sig,
                    order=list(training_data_map.values()))
        ax_sig.set_title('Prediction for significant genes only, performance vs. data type', size=14)
        ax_sig.set_xlabel('Data type', size=14)
        ax_sig.set_ylabel('{}(signal) - {}(shuffled)'.format(
                          metric.upper(), metric.upper()),
                      size=14)
        ax_sig.set_ylim(-0.2, 0.7)
        for tick in ax_sig.get_xticklabels():
            tick.set_fontsize(12)
            tick.set_rotation(30)

    plt.tight_layout()

    if pairwise_tests:
        tests_1_df = add_annotation(ax_all,
                                    results_df,
                                    list(training_data_map.values()),
                                    metric,
                                    pairwise_box_pairs)
        tests_1_df['gene_set'] = 'all'

        if plot_significant:
            tests_2_df = add_annotation(ax_sig,
                                        results_df[results_df.gene.isin(gene_list)],
                                        list(training_data_map.values()),
                                        metric,
                                        pairwise_box_pairs)
            tests_2_df['gene_set'] = 'significant'
        else:
            tests_2_df = pd.DataFrame()

        return pd.concat((tests_1_df, tests_2_df))


def add_annotation(ax, results_df, all_pairs, metric, box_pairs):
    """Add annotation for pairwise statistical tests to box plots."""
    import itertools as it
    from statannot import add_stat_annotation

    # do rank-based tests for all pairs, with Bonferroni correction
    pairwise_tests_df = _pairwise_compare(results_df,
                                          all_pairs,
                                          metric)

    # specify statistical tests to plot
    box_pvals = (pairwise_tests_df
        .set_index(['data_type_1', 'data_type_2'])
        .loc[box_pairs, :]
    ).corr_pval.values

    # only display nearby pairs
    _ = add_stat_annotation(ax,
                            data=results_df.sort_values(by='gene'),
                            x='training_data',
                            y='delta_mean',
                            order=all_pairs,
                            box_pairs=box_pairs,
                            perform_stat_test=False,
                            pvalues=box_pvals,
                            pvalue_thresholds=[(1e-3, '***'),
                                               (1e-2, '**'),
                                               (0.05, '*'),
                                               (1, 'ns')],
                            text_format='star',
                            loc='inside',
                            verbose=0,
                            fontsize=16)

    return pairwise_tests_df


def plot_heatmap(heatmap_df,
                 results_df,
                 different_from_best=True,
                 raw_results_df=None,
                 metric='aupr'):
    """Plot heatmap comparing data types for each gene.

    Arguments
    ---------
    heatmap_df (pd.DataFrame): dataframe with rows as data types, columns as
                               genes, entries are mean AUPR differences
    results_df (pd.DataFrame): dataframe with processed results/p-values
    """
    if different_from_best:
        results_df = get_different_from_best(results_df,
                                             raw_results_df,
                                             metric=metric)

    ax = sns.heatmap(heatmap_df, cmap='Greens',
                     cbar_kws={'aspect': 10, 'fraction': 0.1, 'pad': 0.01})
    ax.xaxis.labelpad = 15

    # outline around heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')

    # outline around colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('{}(signal) - {}(shuffled)'.format(
                       metric.upper(), metric.upper()),
                   labelpad=15)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    ax = plt.gca()

    # add blue highlights to cells that are significant over baseline
    # add red highlights to cells that are significant and "best" predictor for that gene
    if different_from_best:
        for gene_ix, gene in enumerate(heatmap_df.columns):
            for data_ix, data_type in enumerate(heatmap_df.index):
                if (_check_gene_data_type(results_df, gene, data_type) and
                    _check_equal_to_best(results_df, gene, data_type)):
                    ax.add_patch(
                        Rectangle((gene_ix, data_ix), 1, 1, fill=False,
                                  edgecolor='red', lw=3, zorder=1.5)
                    )
                elif _check_gene_data_type(results_df, gene, data_type):
                    ax.add_patch(
                        Rectangle((gene_ix, data_ix), 1, 1, fill=False,
                                  edgecolor='blue', lw=3)
                    )
    else:
        for gene_ix, gene in enumerate(heatmap_df.columns):
            best_data_type = heatmap_df.loc[:, gene].idxmax()
            for data_ix, data_type in enumerate(heatmap_df.index):
                if (best_data_type == data_type) and (
                    _check_gene_data_type(results_df, gene, data_type)):
                    ax.add_patch(
                        Rectangle((gene_ix, data_ix), 1, 1, fill=False,
                                  edgecolor='red', lw=3, zorder=1.5)
                    )
                elif _check_gene_data_type(results_df, gene, data_type):
                    ax.add_patch(
                        Rectangle((gene_ix, data_ix), 1, 1, fill=False,
                                  edgecolor='blue', lw=3)
                    )

    plt.xlabel('Gene name')
    plt.ylabel('Training data type')
    plt.tight_layout()


def get_different_from_best(results_df, raw_results_df, metric='aupr'):
    """Identify best-performing data types for each gene.

    As an alternative to just identifying the data type with the best average
    performance, we want to also identify data types that are "statistically
    equivalent" to the best performer. For each gene, we do the following:

    1) get all data types that significantly outperform the permuted baseline
       ("well-performing" data types)
    2) do pairwise t-tests comparing the best performing data types with
       other well-performing data types
    3) apply an FDR correction for the total number of t-tests

    In each case where the null hypothesis is accepted, we say both data types
    are statistically equivalent. If the null is rejected, the relevant data
    type does not provide statistically equivalent performance to the best
    performing data type.
    """
    from scipy.stats import ttest_rel

    comparison_pvals = []
    for identifier in results_df.gene.unique():
        # compare best with other data types that are significant from
        # baseline, using pairwise t-tests
        # null hypothesis = each pair of results distributions is the same

        # get best data type
        best_data_ix = (
            results_df[results_df.gene == identifier]
              .loc[:, 'delta_mean']
              .idxmax()
        )
        best_data_type = results_df.iloc[best_data_ix, :].training_data

        # get other significant data types
        other_data_types = (
            results_df[(results_df.gene == identifier) &
                       (results_df.training_data != best_data_type) &
                       (results_df.reject_null)]
        )['training_data'].values

        best_data_dist = (
            raw_results_df[(raw_results_df.identifier == identifier) &
                           (raw_results_df.training_data == best_data_type) &
                           (raw_results_df.signal == 'signal') &
                           (raw_results_df.data_type == 'test')]
        ).sort_values(by=['seed', 'fold'])[metric].values

        if len(other_data_types) == 0:
            continue

        for other_data_type in other_data_types:
            # do pairwise t-tests
            other_data_dist = (
                raw_results_df[(raw_results_df.identifier == identifier) &
                               (raw_results_df.training_data == other_data_type) &
                               (raw_results_df.signal == 'signal') &
                               (raw_results_df.data_type == 'test')]
            ).sort_values(by=['seed', 'fold'])[metric].values

            p_value = ttest_rel(best_data_dist, other_data_dist)[1]

            best_id = '{}, {}'.format(identifier, best_data_type)
            other_id = '{}, {}'.format(identifier, other_data_type)

            comparison_pvals.append([identifier, best_data_type,
                                     other_data_type, p_value])

    comparison_df = pd.DataFrame(
        comparison_pvals,
        columns=['gene', 'best_data_type', 'other_data_type', 'p_value']
    )

    # apply multiple testing correction and identify significant similarities
    from statsmodels.stats.multitest import multipletests
    corr = multipletests(comparison_df['p_value'],
                         alpha=0.05,
                         method='fdr_bh')
    comparison_df = comparison_df.assign(corr_pval=corr[1],
                                         reject_null=corr[0])

    # add column to results_df for statistically equal to best
    equal_to_best = []
    for _, vals in results_df.iterrows():
        if not vals['reject_null']:
            equal_to_best.append(False)
        else:
            comp_gene_df = comparison_df[comparison_df.gene == vals['gene']]
            if vals['training_data'] in comp_gene_df.best_data_type.values:
                equal_to_best.append(True)
            elif vals['training_data'] in comp_gene_df.other_data_type.values:
                equal_to_best.append(
                    comp_gene_df[comp_gene_df.other_data_type == vals['training_data']]
                      .reject_null.values[0]
                )
            else:
                # this happens when the data type is the only significant one
                equal_to_best.append(True)

    results_df = results_df.assign(equal_to_best=equal_to_best)
    return results_df


def plot_multi_omics_raw_results(results_df,
                                 axarr,
                                 data_order,
                                 metric='aupr'):

    max_val = results_df[metric].max()

    data_order =['expression.me_27k',
                 'expression.me_450k',
                 'me_27k.me_450k',
                 'expression.me_27k.me_450k']

    # plot mean performance over all genes in Vogelstein dataset
    for ix, gene in enumerate(results_df.identifier.unique()):

        ax = axarr[ix // 3, ix % 3]

        plot_df = results_df[(results_df.identifier == gene) &
                             (results_df.data_type == 'test')]

        sns.boxplot(data=plot_df, x='signal', y=metric, hue='training_data',
                    hue_order=data_order, ax=ax)
        ax.set_title('Prediction for {} mutation'.format(gene))
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.set_ylim(-0.1, max_val)
        ax.legend_.remove()


def plot_multi_omics_results(results_df,
                             axarr,
                             data_names,
                             colors,
                             metric='aupr'):

    delta_metric = 'delta_{}'.format(metric)

    min_aupr = results_df[delta_metric].min()
    max_aupr = results_df[delta_metric].max()

    # plot mean performance over all genes in pilot experiment
    for ix, gene in enumerate(results_df.gene.unique()):

        ax = axarr[ix // 3, ix % 3]

        plot_df = results_df[(results_df.gene == gene)].copy()
        plot_df.training_data.replace(data_names, inplace=True)

        sns.boxplot(data=plot_df, x='training_data', y=delta_metric,
                    order=list(data_names.values()), palette=colors, ax=ax)
        ax.set_title('Prediction for {} mutation'.format(gene), size=13)
        ax.set_xlabel('Training data type', size=13)
        # hide x-axis tick text
        ax.get_xaxis().set_ticklabels([])
        ax.set_ylabel('{}(signal) - {}(shuffled)'.format(
                          metric.upper(), metric.upper()),
                      size=13)
        ax.set_ylim(-0.2, max_aupr)


def plot_best_multi_omics_results(results_df,
                                  ylim=(0, 0.7),
                                  metric='aupr'):

    delta_metric = 'delta_{}'.format(metric)

    from scipy.stats import wilcoxon

    # plot mean performance over all genes in pilot experiment
    plot_df = pd.DataFrame()
    for ix, gene in enumerate(results_df.gene.unique()):

        plot_gene_df = results_df[(results_df.gene == gene)].reset_index(drop=True)

        # get the best-performing data types from the single-omics and multi-omics models
        max_single_data_type = (
            plot_gene_df[plot_gene_df.model_type.str.contains('single-omics')]
              .groupby('training_data')
              .agg('mean')
              [delta_metric].idxmax()
        )
        max_multi_data_type = (
            plot_gene_df[plot_gene_df.model_type.str.contains('multi-omics')]
              .groupby('training_data')
              .agg('mean')
              [delta_metric].idxmax()
        )

        # get samples with that data type
        max_single_df = plot_gene_df[plot_gene_df.training_data == max_single_data_type]
        max_multi_df = plot_gene_df[plot_gene_df.training_data == max_multi_data_type]

        # calculate difference between means and t-test p-val for that data type
        mean_diff = max_single_df[delta_metric].mean() - max_multi_df[delta_metric].mean()
        _, p_val = wilcoxon(max_single_df.sort_values(['seed', 'fold'])[delta_metric].values,
                            max_multi_df.sort_values(['seed', 'fold'])[delta_metric].values)
        print('{} diff: {:.4f} (pval: {:.4f})'.format(gene, mean_diff, p_val))

        plot_df = pd.concat((plot_df, max_single_df, max_multi_df))

    colors = sns.color_palette('Set2')
    sns.boxplot(data=plot_df, x='gene', y=delta_metric, hue='model_type', palette=colors)
    plt.title('Best performing single-omics vs. multi-omics data type, per gene', size=13)
    plt.xlabel('Target gene', size=13)
    plt.ylabel('{}(signal) - {}(shuffled)'.format(
                   metric.upper(), metric.upper()),
               size=13)
    plt.ylim(ylim)
    plt.legend(title='Model type', loc='lower left', fontsize=12, title_fontsize=12)


def _check_gene_data_type(results_df, gene, data_type):
    return results_df[
        (results_df.gene == gene) &
        (results_df.training_data == data_type)].reject_null.values[0]

def _check_equal_to_best(results_df, gene, data_type):
    return results_df[
        (results_df.gene == gene) &
        (results_df.training_data == data_type)].equal_to_best.values[0]


def _label_points(x, y, labels, ax, sig_alpha):
    text_labels = []
    pts = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    for i, point in pts.iterrows():
        if point['y'] > -np.log10(sig_alpha):
            text_labels.append(
                ax.text(point['x'], point['y'], str(point['label']))
            )
    return text_labels


def _label_points_compare(x, y, labels, ax, sig_alpha):
    text_labels = []
    a = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    for i, point in a.iterrows():
        if (
            (point['y'] > -np.log10(sig_alpha)) or
            (point['x'] > 0.1) or
            (abs(point['x']) > 0.2)
        ):
            text_labels.append(
                ax.text(point['x'], point['y'], str(point['label']))
            )
    return text_labels


def _get_overlap_genes(results_df, gene_set, reference='Vogelstein et al.'):
    # start with Vogelstein genes
    vogelstein_genes = set(
        results_df[results_df.training_data == reference]
          .gene.unique()
    )
    if gene_set == reference:
        # plot genes that are in vogelstein AND either of other datasets
        other_genes = set(
            results_df[results_df.training_data != reference]
              .gene.unique()
        )
        overlap_genes = vogelstein_genes.intersection(other_genes)
    else:
        # plot genes that are in this dataset and vogelstein
        other_genes = set(
            results_df[results_df.training_data == gene_set]
              .gene.unique()
        )
        overlap_genes = vogelstein_genes.intersection(other_genes)
    return overlap_genes


def _pairwise_compare(results_df,
                      data_types,
                      metric,
                      correction=True,
                      correction_alpha=0.05,
                      correction_method='bonferroni'):

    import itertools as it
    from scipy.stats import wilcoxon
    p_vals = []
    for dt1, dt2 in it.combinations(data_types, 2):
        r1 = results_df[results_df.training_data == dt1].delta_mean.values
        r2 = results_df[results_df.training_data == dt2].delta_mean.values
        _, p_val = wilcoxon(r1, r2)
        p_vals.append([dt1, dt2, p_val])
    tests_df = pd.DataFrame(p_vals, columns=['data_type_1', 'data_type_2', 'p_value'])
    if correction:
        from statsmodels.stats.multitest import multipletests
        corr = multipletests(tests_df['p_value'],
                             alpha=correction_alpha,
                             method=correction_method)
        tests_df = tests_df.assign(corr_pval=corr[1], reject_null=corr[0])
    return tests_df

