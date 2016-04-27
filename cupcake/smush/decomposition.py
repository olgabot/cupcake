
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from cupcake.smush.base import _ReducedPlotter



class PCAPlotter(_ReducedPlotter):

    axis_label = 'Principal Component {:d}'

    def __init__(self, data, n_components, color, hue, hue_order,
                 palette, saturation, marker, marker_order, text, text_order,
                 linewidth, linewidth_order, edgecolor,
                 edgecolor_order, **pca_kws):
        """Initialize the variables and data for plotting PCA

        Parameters
        ----------
        {input_params}

        """

        pca_kws = {} if pca_kws is None else pca_kws
        pca_kws.setdefault('n_components', n_components)

        self.establish_variables(data)
        self.establish_colors(color, hue, hue_order, palette, saturation)
        self.establish_symbols(marker, marker_order, text, text_order,
                               linewidth, linewidth_order, edgecolor,
                               edgecolor_order)

        self.establish_reducer(PCA, n_components, pca_kws)
        self.compute_reduction()

    def plot(self, ax=None, legend=True, legend_kws=None, title='', **kwargs):
        if ax is None:
            ax = plt.gca()

        if self.groupby is not None:
            grouped = self.plot_data.groupby(self.groupby)
            colors = sns.color_palette(self.palette, n_colors=len(grouped.groups))
#             with sns.color_palette(palette, n_colors=len(grouped.groups)):
            for color, (group, df) in zip(colors, self.plot_data.groupby(self.groupby)):
                marker = group_to_marker[group]
#                 color =
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], 'o',
                        label=group, marker=marker, color=color, **kwargs)
        else:
            ax.plot(self.decomposed.iloc[:, 0], self.decomposed.iloc[:, 1],
                    'o', **kwargs)

        legend_kws = {} if legend_kws is None else legend_kws
        legend_kws.setdefault('loc', 'best')
        if legend:
            ax.legend(**legend_kws)
        explained_variance = self.reducer.explained_variance_ratio_ * 100
        xmin, xmax, ymin, ymax = ax.axis()
        vmax = max(xmax, ymax)
        vmin = min(xmin, ymin)
        vlim = (vmin, vmax)
        ax.set(xlabel='PC1 explains {:.2f}% of variance'.format(explained_variance[0]),
               ylabel='PC2 explains {:.2f}% of variance'.format(explained_variance[1]),
               title=title, xlim=vlim, ylim=vlim)

    def draw_loadings(self, ax):
        pass


# def pcaplot(data, hue_groupby=None, palette=None):

