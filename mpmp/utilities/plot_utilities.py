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
                          identifier='gene',
                          metric='aupr',
                          predict_str='Mutation prediction',
                          xlim=None,
                          ylim=None,
                          label_x_lower_bounds=None,
                          label_y_lower_bounds=None,
                          verbose=False,
                          mark_overlap=False):
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

        data_results_df = (results_df
            .loc[results_df.training_data == training_data, :]
            .copy()
        )

        if mark_overlap:
            overlap_genes = _get_overlap_genes(results_df,
                                               training_data)
            data_results_df['overlap'] = data_results_df.gene.isin(overlap_genes)
            sns.scatterplot(data=data_results_df, x='delta_mean', y='nlog10_p',
                            hue='reject_null', hue_order=[False, True],
                            style='overlap', style_order=[False, True],
                            ax=ax, legend=(ix == 0), s=100)
        else:
            sns.scatterplot(data=data_results_df, x='delta_mean', y='nlog10_p',
                            hue='reject_null', hue_order=[False, True],
                            ax=ax, legend=(ix == 0))

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
            if mark_overlap:
                label_list = [t for t in ax.get_legend_handles_labels()]
                label_list[1][0] = r'Reject $H_0$'
                label_list[1][3] = r'Overlap'
                ax.legend(handles=label_list[0], labels=label_list[1],
                          loc='upper left', fontsize=14, title_fontsize=14)
            else:
                ax.legend(title=r'Reject $H_0$', loc='upper left',
                          fontsize=14, title_fontsize=14)
        ax.set_title(r'{}, {} data'.format(predict_str, training_data), size=14)

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand in inkscape if necessary
        if (label_x_lower_bounds is not None) or (label_y_lower_bounds is not None):
            if label_x_lower_bounds is None:
                label_x_lower_bound = 0
            else:
                label_x_lower_bound = label_x_lower_bounds[ix]
            if label_y_lower_bounds is None:
                label_y_lower_bound = -np.log10(sig_alpha)
            else:
                label_y_lower_bound = label_y_lower_bounds[ix]
            text_labels = _label_points_bound(data_results_df['delta_mean'],
                                              data_results_df['nlog10_p'],
                                              data_results_df[identifier],
                                              ax,
                                              label_x_lower_bound,
                                              label_y_lower_bound)
        else:
            text_labels = _label_points(data_results_df['delta_mean'],
                                        data_results_df['nlog10_p'],
                                        data_results_df[identifier],
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
                            predict_str='Mutation prediction',
                            sig_genes=None,
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

        if sig_genes is not None:
            # get only the training data types involved in this comparison, and
            # identify genes that beat the shuffled baseline for both data types
            sig_genes_comparison = (
                sig_genes[
                    sig_genes.training_data.isin(['gene expression', training_data])
                ]
                  .groupby('gene')
                  .all()
            )['reject_null_baseline']
            # join baseline comparison results into inter-omics comparisons
            compare_results_df = (compare_results_df
                .merge(sig_genes_comparison, on=['gene'])
            )
            # then plot using the baseline results as the marker style,
            # and inter-omics results as the marker hue
            sns.scatterplot(data=compare_results_df, x='delta_mean', y='nlog10_p',
                            hue='reject_null', style='reject_null_baseline',
                            hue_order=[False, True], ax=ax, legend=(ix == 0))
        else:
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
            if sig_genes is not None:
                h, l = ax.get_legend_handles_labels()
                l[0] = r'Reject $H_0$'
                l[3] = r'Reject baseline $H_0$'
                ax.legend(h, l, loc='upper right',
                          fontsize=13, title_fontsize=13)
            else:
                ax.legend(title=r'Reject $H_0$', loc='upper left',
                          fontsize=14, title_fontsize=14)

        ax.set_title(
            r'{}, expression vs. {}'.format(predict_str, training_data),
            size=14
        )

        # label genes and adjust text to not overlap
        # automatic alignment isn't perfect, can align by hand if necessary
        text_labels = _label_points_compare(
                          compare_results_df['delta_mean'],
                          compare_results_df['nlog10_p'],
                          compare_results_df['gene'],
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
                 raw_results_df,
                 metric='aupr',
                 id_name='gene',
                 scale=None,
                 origin_eps_x=0.0,
                 origin_eps_y=0.0,
                 length_x=1.0,
                 length_y=1.0):
    """Plot heatmap comparing data types for each gene.

    Arguments
    ---------
    heatmap_df (pd.DataFrame): dataframe with rows as data types, columns as
                               genes, entries are mean AUPR differences
    results_df (pd.DataFrame): dataframe with processed results/p-values
    """
    # get data types that are equivalent to best-performing data type
    results_df = get_different_from_best(results_df,
                                         raw_results_df,
                                         metric=metric,
                                         id_name=id_name)

    if scale is not None:
        ax = sns.heatmap(heatmap_df, cmap='Greens',
                         cbar_kws={'aspect': 10, 'fraction': 0.1, 'pad': 0.01},
                         vmin=scale[0], vmax=scale[1])
    else:
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

    # add grey dots to cells that are significant over baseline
    # add black dots to cells that are significant and "best" predictor for that gene
    for id_ix, identifier in enumerate(heatmap_df.columns):
        for data_ix, data_type in enumerate(heatmap_df.index):
            if _check_data_type(results_df, identifier, data_type, id_name):
                ax.scatter(id_ix + 0.5, data_ix + 0.5, color='0.8', edgecolors='black', s=200)
            if (_check_data_type(results_df, identifier, data_type, id_name) and
                _check_equal_to_best(results_df, identifier, data_type, id_name)):
                ax.scatter(id_ix + 0.5, data_ix + 0.5, color='black', edgecolor='black', s=60)

    plt.xlabel('{} name'.format(id_name.capitalize().replace('_', ' ')))
    plt.ylabel('Training data type')
    plt.tight_layout()


def get_different_from_best(results_df,
                            raw_results_df,
                            metric='aupr',
                            id_name='gene'):
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
    for identifier in results_df[id_name].unique():
        # compare best with other data types that are significant from
        # baseline, using pairwise t-tests
        # null hypothesis = each pair of results distributions is the same

        # get best data type
        best_data_ix = (
            results_df[results_df[id_name] == identifier]
              .loc[:, 'delta_mean']
              .idxmax()
        )
        best_data_type = results_df.iloc[best_data_ix, :].training_data

        # get other significant data types
        other_data_types = (
            results_df[(results_df[id_name] == identifier) &
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
        columns=[id_name, 'best_data_type', 'other_data_type', 'p_value']
    )

    # apply multiple testing correction and identify significant similarities
    from statsmodels.stats.multitest import multipletests
    corr = multipletests(comparison_df['p_value'],
                         alpha=0.05,
                         method='fdr_bh')
    comparison_df = comparison_df.assign(corr_pval=corr[1],
                                         accept_null=~corr[0])

    # add column to results_df for statistically equal to best
    equal_to_best = []
    for _, vals in results_df.iterrows():
        if not vals['reject_null']:
            equal_to_best.append(False)
        else:
            comp_gene_df = comparison_df[comparison_df[id_name] == vals[id_name]]
            if vals['training_data'] in comp_gene_df.best_data_type.values:
                equal_to_best.append(True)
            elif vals['training_data'] in comp_gene_df.other_data_type.values:
                # reject null = means are significantly different
                # accept null = means are statistically the same
                # so accept null = alternate data type is statistically the
                # same as the best data type
                equal_to_best.append(
                    comp_gene_df[comp_gene_df.other_data_type == vals['training_data']]
                      .accept_null.values[0]
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


def _check_data_type(results_df, identifier, data_type, id_name):
    return results_df[
        (results_df[id_name] == identifier) &
        (results_df.training_data == data_type)].reject_null.values[0]

def _check_equal_to_best(results_df, identifier, data_type, id_name):
    return results_df[
        (results_df[id_name] == identifier) &
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


def _label_points_bound(x, y, labels, ax, x_lower, y_lower):
    text_labels = []
    pts = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    for i, point in pts.iterrows():
        if point['x'] > x_lower and point['y'] > y_lower:
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

