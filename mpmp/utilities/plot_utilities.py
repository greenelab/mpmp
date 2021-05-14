import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

def plot_volcano_baseline(results_df,
                          axarr,
                          training_data_map,
                          sig_alpha,
                          xlim=None,
                          ylim=None,
                          verbose=False):

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


def plot_volcano_comparison():
    pass


def _label_points(x, y, labels, ax, sig_alpha):
    text_labels = []
    pts = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    for i, point in pts.iterrows():
        if point['y'] > -np.log10(sig_alpha):
            text_labels.append(
                ax.text(point['x'], point['y'], str(point['label']))
            )
    return text_labels

