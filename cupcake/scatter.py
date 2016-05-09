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


class PlottingAttribute(object):

    __slots__ = 'groupby', 'title', 'palette'

    def __init__(self, groupby, title, palette):
        """An attribute that you want to visualize with a specific visual cue

        Parameters
        ----------
        groupby : mappable
            A series or dict or list to groupby on the rows of the data
        title : str
            Title of this part of the legend
        palette : list-like
            What to plot for each group
        """
        self.groupby = groupby
        self.title = title
        self.palette = palette


class PlotterMixin(object):

    """
    Must be mixed with something that creates the ``self.plot_data`` attribute


    Attributes
    ----------

    color :



    """
    # Markers that can be filled, in a reasonable order so things that can be
    # confused with each other (e.g. triangles pointing to the left or right) are
    # not next to each other
    filled_markers = (u'o', u'v', u's', u'*', u'h', u'<', u'H', u'x', u'8',
                      u'>', u'D', u'd', u'^')
    linewidth_min, linewidth_max = 0.1, 5

    alpha_min, alpha_max = 0.1, 1

    size_min, size_max = 3, 30

    legend_order = 'color', 'symbol', 'linewidth', 'edgecolor', 'alpha', 'size'

    def establish_colors(self, color, hue, hue_order, palette):
        """Get a list of colors for the main component of the plots."""
        n_colors = None

        current_palette = sns.utils.get_color_cycle()


        color_labels = None
        color_title = None


        if color is not None and palette is not None:
            error = 'Cannot interpret colors to plot when both "color" and ' \
                    '"palette" are specified'
            raise ValueError(error)


        # Force "hue" to be a mappable
        if hue is not None:
            try:
                # Check if "hue" is a column in the data
                color_title = str(hue)
                hue = self.data[hue]
            except (ValueError, KeyError):
                # Hue is already a mappable
                if isinstance(hue, pd.Series):
                    color_title = hue.name
                else:
                    color_title = None

            # This will give the proper number of categories even if there are
            #  more categories in "hue_order" than represented in "hue"
            hue_order = sns.utils.categorical_order(hue, hue_order)
            color_labels = hue_order
            hue = pd.Categorical(hue, hue_order)
            n_colors = len(self.plot_data.groupby(hue))
        else:
            if hue_order is not None:

                # Check if "hue_order" specifies rows in the data
                samples_to_plot = self.plot_data.index.intersection(hue_order)
                n_colors = len(samples_to_plot)
                if n_colors > 0:
                    # Different color for every sample (row name)
                    hue = pd.Series(self.plot_data.index,
                                    index=self.plot_data.index)

                else:
                    error = "When 'hue=None' and 'hue_order' is specified, " \
                            "'hue_order' must overlap with the data row " \
                            "names (index)"
                    raise ValueError(error)
            else:
                # Same color for everything
                hue = pd.Series('hue', index=self.plot_data.index)
                n_colors = 1

        if palette is not None:
            colors = sns.color_palette(palette, n_colors=n_colors)
        elif color is not None:
            colors = sns.light_palette(color, n_colors=n_colors)
        else:
            colors = sns.light_palette(current_palette[0],
                                       n_colors=n_colors)
        self.color = PlottingAttribute(hue, color_title, colors)

    def _maybe_make_grouper(self, attribute, palette_maker, order=None,
                            func=None, default=None):
        """Create a Series from a single attribute, else make categorical

        Checks if the attribute is in the data provided, or is an external
        mapper

        Parameters
        ----------
        attribute : object
            Either a single item to create into a series, or a series mapping
            each sample to an attribute (e.g. the plotting symbol 'o' or
            linewidth 1)
        palette_maker : function
            Function which takes an integer and creates the appropriate
            palette for the attribute, e.g. shades of grey for edgecolor or
            linearly spaced sizes
        order : list
            The order to create the attributes into
        func : function
            A function which returns true if the attribute is a single valid
            instance, e.g. "black" for color or 0.1 for linewidth. Otherwise,
            we assume that "attribute" is a mappable

        Returns
        -------
        grouper : pandas.Series
            A mapping of the high dimensional data samples to the attribute
        """

        title = None

        if func is None or func(attribute):
            # Use this single attribute for everything
            return PlottingAttribute(pd.Series(None, index=self.samples),
                                     title, (attribute,))
        else:

            try:
                # Check if this is a column in the data
                attribute = self.data[attribute]
            except (ValueError, KeyError):
                pass

            if isinstance(attribute, pd.Series):
                title = attribute.name
            order = sns.utils.categorical_order(attribute, order)

            palette = palette_maker(len(order))
            attribute = pd.Categorical(attribute, categories=order,
                                       ordered=True)
            return PlottingAttribute(pd.Series(attribute, index=self.samples),
                                     title, palette)

    def establish_symbols(self, marker, marker_order, text, text_order):
        """Figure out what symbol put on the axes for each data point"""

        symbol_title = None

        if isinstance(text, bool):
            # Option 1: Text is a boolean
            if text:
                # 1a: text=True, so use the sample names of data as the
                # plotting symbol
                symbol_title = 'Samples'
                symbols = [str(x) for x in self.samples]
                symbol = pd.Series(self.samples, index=self.samples)
            else:
                # 1b: text=False, so use the specified marker for each sample
                symbol = self._maybe_make_grouper(marker, marker_order, str)
                if marker is not None:
                    try:
                        symbol_title = marker
                        symbol = self.data[marker]
                        symbols = sns.categorical_order(symbol, marker_order)
                    except (ValueError, KeyError):
                        # Marker is a single marker, or already a groupable
                        if marker in self.filled_markers:
                            # Single marker so make a tuple so it's indexable
                            symbols = (marker,)
                        else:
                            # already a groupable object
                            if isinstance(marker, pd.Series):
                                symbol_title = marker.name
                            n_symbols = len(self.plot_data.groupby(symbol))
                            if n_symbols > len(self.filled_markers):
                                # If there's too many categories, then
                                # auto-expand the existing list of filled
                                # markers
                                multiplier = np.ceil(
                                    n_symbols/float(len(self.filled_markers)))
                                filled_markers = list(self.filled_markers) \
                                                 * multiplier
                                symbols = filled_markers[:n_symbols]
                            else:
                                symbols = self.filled_markers[:n_symbols]
        else:
            # Assume "text" is a mapping from row names (sample ids) of the
            # data to text labels
            text_order = sns.utils.categorical_order(text, text_order)
            symbols = text_order
            symbol = pd.Series(pd.Categorical(text, categories=text_order,
                                              ordered=True),
                               index=self.samples)
            if marker is not None:
                warnings.warn('Overriding plotting symbol from "marker" with '
                              'values in "text"')

            # Turn text into a boolean
            text = True

        self.symbol = PlottingAttribute(symbol, symbol_title, symbols)

        self.text = text

    def establish_symbol_attributes(self,linewidth, linewidth_order, edgecolor,
                                    edgecolor_order, alpha, alpha_order, size,
                                    size_order):
        self.edgecolor = self._maybe_make_grouper(
            edgecolor, self._edgecolor_palette, edgecolor_order,
            mpl.colors.is_color_like)
        self.linewidth = self._maybe_make_grouper(
            linewidth, self._linewidth_palette, linewidth_order, np.isfinite)
        self.alpha = self._maybe_make_grouper(
            alpha, self._alpha_palette, alpha_order, np.isfinite)
        self.size = self._maybe_make_grouper(
            size, self._size_palette, size_order, np.isfinite)

    @staticmethod
    def _edgecolor_palette(self, n_groups):
        return sns.color_palette('Greys', n_colors=n_groups)

    def _linewidth_palette(self, n_groups):
        return np.linspace(self.linewidth_min, self.linewidth_max, n_groups)

    def _alpha_palette(self, n_groups):
        return np.linspace(self.alpha_min, self.alpha_max, n_groups)

    def _size_palette(self, n_groups):
        return np.linspace(self.size_min, self.size_max, n_groups)

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
            # Add dummy plot to make the axes in the right window
            ax.plot(xs, ys, color=None)
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
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)

    def establish_legend_data(self):
        self.legend_data = pd.DataFrame(dict(color=self.color.groupby,
                                             symbol=self.symbol.groupby,
                                             size=self.size.groupby,
                                             linewidth=self.linewidth.groupby,
                                             edgecolor=self.edgecolor.groupby,
                                             alpha=self.alpha.groupby))
        self.legend_data = self.legend_data.reindex(columns=self.legend_order)

    def draw_symbols(self, ax, plot_kws):
        """Plot each sample in the data"""

        # Iterate over all the possible modifications of the points
        # TODO: add alpha and size
        for i, (color_label, df1) in enumerate(self.plot_data.groupby(self.color.groupby)):
            color = self.color.palette[i]
            for j, (marker_label, df2) in enumerate(df1.groupby(self.symbol.groupby)):
                symbol = self.symbol.palette[j]
                for k, (lw_label, df3) in enumerate(df2.groupby(self.linewidth.groupby)):
                    linewidth = self.linewidth.palette[k]
                    for l, (ec_label, df4) in df3.groupby(self.edgecolor):
                        edgecolor = self.edgecolor.palette[l]
                        # and finally ... actually plot the data!
                        for m
                        self.symbolplotter(df4.iloc[:, 0], df4.iloc[:, 1],
                                           symbol=symbol, color=color,
                                           ax=ax, linewidth=linewidth,
                                           edgecolor=edgecolor, **plot_kws)



class ScatterPlotter(PlotterMixin):

    def __init__(self, data, x, y, color, hue, hue_order, palette, marker,
                 marker_order, text, text_order, linewidth, linewidth_order,
                 edgecolor, edgecolor_order, alpha, alpha_order, size,
                 size_order):
        self.establish_data(data, x, y)
        self.establish_symbols(marker, marker_order, text, text_order)
        self.establish_symbol_attributes(linewidth, linewidth_order, edgecolor,
            edgecolor_order, alpha, alpha_order, size, size_order)
        self.establish_colors(color, hue, hue_order, palette)
        self.establish_legend_data()
        # import pdb; pdb.set_trace()

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
        self.draw_symbols(ax, kwargs)
        self.annotate_axes(ax)


def scatterplot(data, x=0, y=1, color=None, hue=None, hue_order=None,
                palette=None, marker='o', marker_order=None, text=False,
                text_order=None, linewidth=1, linewidth_order=None,
                edgecolor='k', edgecolor_order=None, alpha=1, alpha_order=None,
                size=7, size_order=None, ax=None, **kwargs):

    plotter = ScatterPlotter(data, x, y, color, hue, hue_order, palette,
                             marker, marker_order, text, text_order, linewidth,
                             linewidth_order, edgecolor, edgecolor_order,
                             alpha, alpha_order, size, size_order)
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax
