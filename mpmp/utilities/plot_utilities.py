import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

import mpmp.utilities.analysis_utilities as au

def plot_volcano_baseline(results_df,
                          axarr,
                          training_data_map,
                          sig_alpha,
                          xlim=None,
                          ylim=None,
                          verbose=False):
    """Make a scatter plot comparing classifier results to shuffled baseline.

    Arguments
    ---------
    results_df (pd.DataFrame): dataframe with processed results/p-values
    axarr (array of Axes): axes array from plt.subplots
    training_data_map (dict): maps internal data type names to display names
    sig_alpha (float): significance testing threshold

    Returns
    -------
    axarr (array of Axes): axes with plots
    """

    # set axis limits if not provided
    if xlim is None:
        xlim = (-0.2, 1.0)
    if ylim is None:
        y_max = results_df.nlog10_p.max()
        ylim = (0, y_max+3)

    # plot mutation prediction from expression, in a volcano-like plot
    for ix, training_data in enumerate(training_data_map.values()):

        if axarr.ndim > 1:
            ax = axarr[ix // axarr.shape[1], ix % axarr.shape[1]]
        else:
            ax = axarr[ix]

        data_results_df = results_df[results_df.training_data == training_data]

        sns.scatterplot(data=data_results_df, x='delta_mean', y='nlog10_p', hue='reject_null',
                        hue_order=[False, True], ax=ax, legend=(ix == 0))

        # add vertical line at 0
        ax.axvline(x=0, linestyle='--', linewidth=1.25, color='black')

        # add horizontal line at statistical significance threshold
        l = ax.axhline(y=-np.log10(sig_alpha), linestyle='--', linewidth=1.25)

        # label horizontal line with significance threshold
        # (matplotlib makes this fairly difficult, sadly)
        ax.text(0.9, -np.log10(sig_alpha)+0.02,
                r'$\mathbf{{\alpha = {}}}$'.format(sig_alpha),
                va='center', ha='center', color=l.get_color(),
                backgroundcolor=ax.get_facecolor())

        # label axes and set axis limits
        ax.set_xlabel('AUPR(signal) - AUPR(shuffled)', size=14)
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

    return axarr


def plot_volcano_comparison(results_df,
                            axarr,
                            training_data_map,
                            sig_alpha,
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

    Returns
    -------
    axarr (array of Axes): axes with plots
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

        if axarr.ndim > 1:
            ax = axarr[ix // axarr.shape[1], ix % axarr.shape[1]]
        else:
            ax = axarr[ix]

        data_results_df = results_df[results_df.training_data == training_data].copy()
        data_results_df.sort_values(by=['seed', 'fold'], inplace=True)
        compare_results_df = au.compare_results(exp_results_df,
                                                condition_2_df=data_results_df,
                                                identifier='identifier',
                                                metric='aupr',
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

        # add horizontal line at statistical significance threshold
        l = ax.axhline(y=-np.log10(sig_alpha), linestyle='--', linewidth=1.25)

        # label horizontal line with significance threshold
        # (matplotlib makes this fairly difficult, sadly)
        ax.text(0.5, -np.log10(sig_alpha)+0.01,
                r'$\mathbf{{\alpha = {}}}$'.format(sig_alpha),
                va='center', ha='center', color=l.get_color(),
                backgroundcolor=ax.get_facecolor())

        # label axes and set axis limits
        ax.set_xlabel('AUPR({}) - AUPR(expression)'.format(training_data), size=14)
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

    return axarr


def plot_boxes(results_df,
               axarr,
               training_data_map,
               orientation='h',
               verbose=False):
    """Make a box plot comparing classifier results between data types.

    Arguments
    ---------
    results_df (pd.DataFrame): dataframe with processed results/p-values
    axarr (array of Axes): axes array from plt.subplots
    training_data_map (dict): maps internal data type names to display names

    Returns
    -------
    axarr (array of Axes): axes with plots
    """

    # plot mean performance over all genes in Vogelstein dataset
    ax = axarr[0]
    sns.boxplot(data=results_df, x='training_data', y='delta_mean', notch=True,
                ax=ax, order=list(training_data_map.values()))
    ax.set_title('Prediction for all genes, performance vs. data type', size=14)
    if orientation == 'v':
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Data type', size=14)
    ax.set_ylabel('AUPR(signal) - AUPR(shuffled)', size=14)
    ax.set_ylim(-0.2, 0.7)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_rotation(30)

    # plot mean performance for genes that are significant for at least one data type
    ax = axarr[1]
    gene_list = results_df[results_df.reject_null == True].gene.unique()
    if verbose:
        print(gene_list.shape)
        print(gene_list)
    sns.boxplot(data=results_df[results_df.gene.isin(gene_list)],
                x='training_data', y='delta_mean', notch=True, ax=ax,
                order=list(training_data_map.values()))
    ax.set_title('Prediction for significant genes only, performance vs. data type', size=14)
    ax.set_xlabel('Data type', size=14)
    ax.set_ylabel('AUPR(signal) - AUPR(shuffled)', size=14)
    ax.set_ylim(-0.2, 0.7)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_rotation(30)

    plt.tight_layout()


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

