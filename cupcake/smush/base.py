"""
Internal abstract base class for all dimensionality reduction plots.
Not intended to be user-facing.
"""
from __future__ import division
import colorsys
import matplotlib as mpl
from textwrap import dedent
import warnings

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import pandas as pd
import seaborn as sns


class _ReducedPlotter(object):
    """Generic object for plotting high-dimensional data on 2d space"""

    def establish_reducer(self, reducer, n_components=None, smusher_kws=None):

        try:
            # Create a brand new dimensionality reducer
            # (matrix decomposer/manifold learner)

            smusher_kws = {} if smusher_kws is None else smusher_kws
            smusher_kws.setdefault('n_components', n_components)

            self.reducer = reducer(**smusher_kws)
        except TypeError:
            # We're using a pre-existing, user-supplied reducer
            self.reducer = reducer

    def establish_variables(self, data):
        """Convert input specification into a common representation."""

        # Option 1a:
        # The input data is a Pandas DataFrame
        # ------------------------------------
        if isinstance(data, pd.DataFrame):
            high_dimensional_data = data
            group_label = data.index.name
            value_label = data.columns.name

        # Option 1b:
        # The input data is an array or list
        # ----------------------------------
        else:
            if len(data.shape) == 2:
                nr, nc = data.shape
                if nr == 1 or nc == 1:
                    error = ("Input `data` must have "
                             "exactly 2 dimensions, where both "
                             "have at least 2 components")
                    raise ValueError(error)
                else:
                    high_dimensional_data = pd.DataFrame(data)
            else:
                error = ("Input `data` must have "
                         "exactly 2 dimensions")
                raise ValueError(error)

            group_label = None
            value_label = None

        # Assign object attributes
        # ------------------------
        self.high_dimensional_data = high_dimensional_data
        self.group_label = group_label
        self.value_label = value_label

    def establish_colors(self, color, palette, saturation):
        """Get a list of colors for the main component of the plots."""
        if self.hue_names is None:
            n_colors = len(self.reduced_data.index)
        else:
            n_colors = len(self.hue_names)

        # Determine the main colors
        if color is None and palette is None:
            # Determine whether the current palette will have enough values
            # If not, we'll default to the husl palette so each is distinct
            current_palette = sns.utils.get_color_cycle()
            if n_colors <= len(current_palette):
                colors = sns.color_palette(n_colors=n_colors)
            else:
                colors = sns.husl_palette(n_colors, l=.7)

        elif palette is None:
            # When passing a specific color, the interpretation depends
            # on whether there is a hue variable or not.
            # If so, we will make a blend palette so that the different
            # levels have some amount of variation.
            if self.hue_names is None:
                colors = [color] * n_colors
            else:
                colors = sns.light_palette(color, n_colors)
        else:

            # Let `palette` be a dict mapping level to color
            if isinstance(palette, dict):
                if self.hue_names is None:
                    levels = self.group_names
                else:
                    levels = self.hue_names
                palette = [palette[l] for l in levels]

            colors = sns.color_palette(palette, n_colors)

        # Desaturate a bit because these are patches
        if saturation < 1:
            colors = sns.color_palette(colors, desat=saturation)

        # Convert the colors to a common representations
        rgb_colors = sns.color_palette(colors)

        # Determine the gray color to use for the lines framing the plot
        light_vals = [colorsys.rgb_to_hls(*c)[1] for c in rgb_colors]
        l = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((l, l, l))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray


    def _maybe_make_grouper(self, attribute, attribute_order, dtype):
        if isinstance(attribute, dtype):
            # Use this single attribute for everything
            return pd.Series([attribute]*self.high_dimensional_data.shape[0],
                             index=self.high_dimensional_data.index)
        else:
            attribute_order = sns.utils.categorical_order(attribute,
                                                          attribute_order)
            return pd.Categorical(attribute, categories=attribute_order,
                                  ordered=True)

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
                    map(str, self.high_dimensional_data.index))
            else:
                # 1b: text=False, so use the specified marker for each sample
                symbol = self._maybe_make_grouper(marker, marker_order, str)

        else:
            # Assume "text" is a mapping from row names (sample ids) of the
            # data to text labels
            text_order = sns.utils.categorical_order(text, text_order)
            symbol = pd.Categorical(text, categories=text_order, ordered=True)
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

    def symbolplotter(self, xs, ys, symbol, linewidth, edgecolor, **kwargs):
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
                plt.text(x, y, symbol, **kwargs)
        else:
            # use plt.plot instead of plt.scatter for speed, since plotting all
            # the same marker shape and color and linestyle
            plt.plot(xs, ys, 'o', marker=symbol, linewidth=linewidth,
                     edgecolor=edgecolor, **kwargs)

    def annotate_axes(self, ax):
        """Add descriptive labels to an Axes object."""
        if self.orient == "v":
            xlabel, ylabel = self.group_label, self.value_label
        else:
            xlabel, ylabel = self.value_label, self.group_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if self.orient == "v":
            ax.set_xticks(np.arange(len(self.reduced_data)))
            ax.set_xticklabels(self.group_names)
        else:
            ax.set_yticks(np.arange(len(self.reduced_data)))
            ax.set_yticklabels(self.group_names)

        if self.orient == "v":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, len(self.reduced_data) - .5)
        else:
            ax.yaxis.grid(False)
            ax.set_ylim(-.5, len(self.reduced_data) - .5)

        if self.hue_names is not None:
            leg = ax.legend(loc="best")
            if self.hue_title is not None:
                leg.set_title(self.hue_title)

                # Set the title size a roundabout way to maintain
                # compatability with matplotlib 1.1
                try:
                    title_size = mpl.rcParams["axes.labelsize"] * .85
                except TypeError:  # labelsize is something like "large"
                    title_size = mpl.rcParams["axes.labelsize"]
                prop = mpl.font_manager.FontProperties(size=title_size)
                leg._legend_title_box._text.set_font_properties(prop)

    def add_legend_data(self, ax, color, label):
        """Add a dummy patch object so we can get legend data."""
        rect = plt.Rectangle([0, 0], 0, 0,
                             linewidth=self.linewidth / 2,
                             edgecolor=self.gray,
                             facecolor=color,
                             label=label)
        ax.add_patch(rect)

    def draw_markers(self, ax, x=0, y=1, **kwargs):
        """Plot each sample in the data in the reduced space"""

        # Iterate over all the possible modifications of the points
        for hue, hue_df in self.reduced_data.groupby(self.hue):
            for symbol, symbol_df in hue_df.groupby(self.symbol):
                for lw, lw_df in symbol_df.groupby(self.linewidth):
                    for ec, ec_df in lw_df.groupby(self.edgecolor):
                        # and finally ... actually plot the data!
                        self.symbolplotter(ec_df[:, x], ec_df[:, y],
                                           symbol=symbol, color=hue,
                                           ax=ax, linewidth=lw, edgecolor=ec,
                                           **kwargs)

    def draw_images(self, ax, images):
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(self.reduced_data.shape[0]):
                dist = np.sum((self.reduced_data[i] - shown_images) ** 2, 1)
                if np.min(dist) < 5e-4 * np.product(images[0].shape):
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [self.reduced_data[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=plt.cm.gray),
                    self.reduced_data[i])
                ax.add_artist(imagebox)

    def compute_reduction(self):
        """Use reducer to smush high-dimensional data into a smaller space"""
        matrix = self.reducer.fit_transform(self.high_dimensional_data)
        columns = np.arange(matrix.shape[1], dtype=int) + 1
        self.reduced_data = pd.DataFrame(
            matrix, columns=columns, index=self.high_dimensional_data.index)


_reducer_docs = dict(
    input_params=dedent("""
    data : DataFrame, array, or list of arrays
        Dataset to reduce dimensionality and plot.
        """),
    component_input_params=dedent("""\
    x, y : int
        1-based integer of components to plot on "x" and "y" axes. Defaults
        are ``x=1``, ``y=2``.\
        """),

    groupby_input_params=dedent("""\
    hue, marker : mapping (dict or pandas.Series)
        A dict or pandas.Series mapping the row names of ``data`` to a
        separate category to plot as separate colors (hue) or plotting
        symbols (marker). See examples for interpretation.\
        """),

    # From seaborn.categorical
    color=dedent("""\
    color : matplotlib color, optional
        Color for all of the elements, or seed for :func:`light_palette` when
        using hue nesting.\
    """),
    palette=dedent("""\
    palette : palette name, list, or dict, optional
        Color palette that maps either the grouping variable or the hue
        variable. If the palette is a dictionary, keys should be names of
        levels and values should be matplotlib colors.\
    """),
    saturation=dedent("""\
    saturation : float, optional
        Proportion of the original saturation to draw colors at. Large patches
        often look better with slightly desaturated colors, but set this to
        ``1`` if you want the plot colors to perfectly match the input color
        spec.\
    """),
    ax_in=dedent("""\
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
)
