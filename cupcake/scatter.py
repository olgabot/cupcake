"""
Generic, configurable scatterplot
"""
import itertools
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotterMixin(object):

    """
    Must be mixed with something that creates the ``self.plot_data`` attribute
    """

    def establish_colors(self, color, hue, hue_order, palette):
        """Get a list of colors for the main component of the plots."""
        n_colors = None

        if color is None:
            # No color_groupby is specified
            if palette is None:
                # Auto-assigned palette
                if hue is None:
                    current_palette = sns.utils.get_color_cycle()
                    color = current_palette[0]
                    if hue_order is None:
                        # Auto-assign one color_groupby to all samples from
                        # current color cycle
                        n_colors = 1
                        color_groupby = self._maybe_make_grouper(color)
                    else:
                        if len(hue_order) == self.n_samples:
                            n_colors = self.n_samples
                            colors = sns.light_palette(color,
                                                       n_colors=n_colors)
                            color_groupby = pd.Series(colors,
                                                      index=self.samples)
                        else:
                            error = 'Cannot interpret "hue_order" when "hue"' \
                                    ' is not specified [or len(hue_order) !=' \
                                    ' len(data.index)]'
                            raise ValueError(error)

                else:
                    # Use the hue grouping, possibly with an order (handled
                    # within _maybe_make_grouper)
                    color_groupby = self._maybe_make_grouper(hue,
                                                             order=hue_order,
                                                             dtype=str)
                    n_colors = len(self.plot_data.groupby(
                        color_groupby).size())
            else:
                # User-defined palette
                if hue is None:
                    # Assign every sample a different color_groupby from
                    # palette
                    n_colors = self.n_samples
                    colors = sns.color_palette(palette, n_colors=n_colors)

                    index = self.samples if hue_order is None else hue_order
                    color_groupby = pd.Series(colors, index=index)
                else:
                    # Assign every group a different color_groupby from palette

                    grouped = self.plot_data.groupby(hue)
                    size = grouped.size()
                    n_colors = size.shape[0]
                    palette = sns.color_palette(palette, n_colors=n_colors)

                    color_groupby = self._color_grouper_from_palette(
                        grouped, palette, hue_order)
        else:
            # Single color_groupby is provided
            if palette is None:
                if hue is None:
                    if hue_order is None:
                        color_groupby = self._maybe_make_grouper(color)
                        n_colors = 1
                    else:
                        if len(hue_order) == self.n_samples:
                            n_colors = self.n_samples
                            colors = sns.light_palette(color,
                                                       n_colors=n_colors)
                            color_groupby = pd.Series(colors,
                                                      index=self.samples)
                        else:
                            error = 'Cannot interpret "hue_order" when "hue"' \
                                    ' is not specified [or len(hue_order) !=' \
                                    ' len(data.index)]'
                            raise ValueError(error)
                else:
                    grouped = self.plot_data.groupby(hue)
                    size = grouped.size()
                    self.n_colors = len(size)
                    palette = sns.light_palette(color, n_colors=n_colors)

                    color_groupby = self._color_grouper_from_palette(
                        grouped, palette, hue_order)
            else:
                error = 'Cannot specify both "palette" and "color"'
                raise ValueError(error)

        # Assign object attributes
        self.n_colors = n_colors
        self.color = color_groupby

    def _maybe_make_grouper(self, attribute, order=None, dtype=None):
        """Create a Series from a single attribute, else make categorical

        Parameters
        ----------
        attribute : object
            Either a single item to create into a series, or a series mapping
            each sample to an attribute (e.g. the plotting symbol 'o' or
            linewidth 1)
        order : list
            The order to create the attributes into
        dtype : type
            If "attribute" is of this type (as in, it is a single item), then
            create a single series. This is for consistency so that every
            possible plotting attribute can be used in a "groupby"

        Returns
        -------
        grouper : pandas.Series
            A mapping of the high dimensional data samples to the attribute
        """
        if dtype is None or isinstance(attribute, dtype):
            # Use this single attribute for everything
            return pd.Series([attribute]*self.n_samples, index=self.samples)
        else:
            order = sns.utils.categorical_order(attribute, order)
            attribute = pd.Categorical(attribute, categories=order,
                                       ordered=True)
            return pd.Series(attribute, index=self.samples)

    def _color_grouper_from_palette(self, grouped, palette, hue_order):
        color_tuples = [zip(df.index, [palette[i]] * df.shape[0])
                        for i, (g, df) in enumerate(grouped)]

        colors = dict(itertools.chain(*color_tuples))
        color_groupby = self._maybe_make_grouper(colors, hue_order)
        return color_groupby

    def establish_symbols(self, marker, marker_order, text, text_order,
                          linewidth, linewidth_order, edgecolor,
                          edgecolor_order):
        """Figure out what to put on the axes for each sample in the data"""
        if isinstance(text, bool):
            # Option 1: Text is a boolean
            if text:
                # 1a: text=True, so use the sample names of data as the
                # plotting symbol
                symbol = pd.Series(
                    map(str, self.samples), index=self.samples)
            else:
                # 1b: text=False, so use the specified marker for each sample
                symbol = self._maybe_make_grouper(marker, marker_order, str)

        else:
            # Assume "text" is a mapping from row names (sample ids) of the
            # data to text labels
            text_order = sns.utils.categorical_order(text, text_order)
            symbol = pd.Series(pd.Categorical(text, categories=text_order,
                                              ordered=True),
                               index=self.samples)
            if marker is not None:
                warnings.warn('Overriding plotting symbol from "marker" with '
                              'values in "text"')

            # Turn text into a boolean
            text = True

        self.symbol = symbol

        # Set these even if text=True because they won't be used when plotting
        # the text anyways
        self.linewidth = self._maybe_make_grouper(linewidth, linewidth_order,
                                                  (int, float))
        self.edgecolor = self._maybe_make_grouper(edgecolor, edgecolor_order,
                                                  str)
        self.text = text

    def symbolplotter(self, xs, ys, ax, symbol, linewidth, edgecolor, **kwargs):
        """Plots either a matplotlib marker or a string at each data position

        Wraps plt.text and plt.plot

        Parameters
        ----------
        xs : array-like
            List of x positions for data
        ys : array-like
            List of y-positions for data
        symbol : str
            What to plot at each (x, y) data position
        text : bool
            If true, then "symboL" is assumed to be a string and iterates over
            each data point individually, using plt.text to position the text.
            Otherwise, "symbol" is a matplotlib marker and uses plt.plot for
            plotting
        kwargs
            Any other keyword arguments to plt.text or plt.plot
        """
        # If both the x- and y- positions don't have data, don't do anything
        if xs.empty and ys.empty:
            return

        if self.text:
            # Plot each (x, y) position as text
            for x, y in zip(xs, ys):
                ax.text(x, y, symbol, **kwargs)
        else:
            # use plt.plot instead of plt.scatter for speed, since plotting all
            # the same marker shape and color and linestyle
            ax.plot(xs, ys, 'o', marker=symbol, markeredgewidth=linewidth,
                     markeredgecolor=edgecolor, **kwargs)

    def annotate_axes(self, ax):
        """Add descriptive labels to an Axes object."""
        xlabel, ylabel = self.xlabel, self.ylabel

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)


    def add_legend_data(self, ax, color, label):
        """Add a dummy patch object so we can get legend data."""
        rect = plt.Rectangle([0, 0], 0, 0,
                             linewidth=self.linewidth / 2,
                             edgecolor=self.gray,
                             facecolor=color,
                             label=label)
        ax.add_patch(rect)

    def draw_markers(self, ax, plot_kws):
        """Plot each sample in the data in the reduced space"""

        # Iterate over all the possible modifications of the points
        # TODO: add alpha and size
        for color, color_df in self.plot_data.groupby(self.color):
            for symbol, symbol_df in color_df.groupby(self.symbol):
                for lw, lw_df in symbol_df.groupby(self.linewidth):
                    for ec, ec_df in lw_df.groupby(self.edgecolor):
                        # and finally ... actually plot the data!
                        self.symbolplotter(ec_df.iloc[:, 0], ec_df.iloc[:, 1],
                                           symbol=symbol, color=color,
                                           ax=ax, linewidth=lw, edgecolor=ec,
                                           **plot_kws)



class ScatterPlotter(PlotterMixin):

    def __init__(self, data, x, y, color, hue, hue_order, palette, marker,
                 marker_order, text, text_order, linewidth, linewidth_order,
                 edgecolor, edgecolor_order):
        self.establish_data(data, x, y)
        self.establish_symbols(marker, marker_order, text, text_order,
                               linewidth, linewidth_order, edgecolor,
                               edgecolor_order)
        self.establish_colors(color, hue, hue_order, palette)

    def establish_data(self, data, x, y):
        if isinstance(data, pd.DataFrame):
            xlabel = data.columns[x]
            ylabel = data.columns[y]
        else:
            data = pd.DataFrame(data)
            xlabel = None
            ylabel = None

        self.data = data
        self.plot_data = self.data.iloc[:, [x, y]]
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.samples = self.plot_data.index
        self.features = self.plot_data.columns
        self.n_samples = len(self.samples)
        self.n_features = len(self.features)

    def plot(self, ax, kwargs):
        self.draw_markers(ax, kwargs)
        self.annotate_axes(ax)


def scatterplot(data, x=0, y=1, color=None, hue=None, hue_order=None,
                palette=None, marker='o', marker_order=None, text=False,
                text_order=None, linewidth=1, linewidth_order=None,
                edgecolor='k', edgecolor_order=None, ax=None, **kwargs):

    plotter = ScatterPlotter(data, x, y, color, hue, hue_order, palette,
                             marker, marker_order, text, text_order, linewidth,
                             linewidth_order, edgecolor, edgecolor_order)
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax
