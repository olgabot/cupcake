"""
Internal abstract base class for all dimensionality reduction plots.
Not intended to be user-facing.
"""
from __future__ import division
from textwrap import dedent
import colorsys
import numpy as np
from scipy import stats
import pandas as pd
from pandas.core.series import remove_na
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import pandas as pd
from scipy.spatial import distance
import seaborn as sns


class _ReducedPlotter(object):
    """Generic object for plotting reduced representations of high-dimensional data on 2d space"""

    def establish_reducer(self, reducer, n_components, smusher_kws):
        smusher_kws.setdefault('n_components', n_components)

        if reducer is None:
            # Create a brand new dimensionality reducer (matrix decomposer/manifold learner)
            self.reducer = reducer(**smusher_kws)
        else:
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

        # Conver the colors to a common representations
        rgb_colors = sns.color_palette(colors)

        # Determine the gray color to use for the lines framing the plot
        light_vals = [colorsys.rgb_to_hls(*c)[1] for c in rgb_colors]
        l = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((l, l, l))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray

    def establish_symbols(self, marker, marker_order, text, text_order):
        """Figure out what to place on the axes for each data point"""
        if isinstance(text, bool):
            self.text = text

            if self.text:
                # Use the sample names of data as the plotting symbol
                symbol = sns.utils.categorical_order(
                    self.high_dimensional_data.index)
            else:
                symbol = sns.utils.categorical_order(marker, marker_order)
        else:
            # Assume "text" is a mapping from row names (sample ids) of the
            # data to text labels
            if marker is not None:
                symbol = sns.utils.categorical_order(text, text_order)
            else:
                error = 'Cannot specify both "marker" and "text'
                raise ValueError(error)

            # Turn text into a boolean
            text = True

        self.symbol = symbol
        self.text = text


    def markerplotter(self, xs, ys, symbol, text, **kwargs):
        # If both the x- and y- positions don't have data, don't do anything
        if xs.empty and ys.empty:
            return

        if text:
            # Plot each (x, y) position as text
            for x, y in zip(xs, ys):
                plt.text(x, y, symbol, **kwargs)
        else:
            # use plt.plot instead of plt.scatter for speed, since plotting all
            # the same marker shape and color and linestyle
            plt.plot(xs, ys, 'o', marker=symbol, **kwargs)

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

    def draw_markers(self, ax, text, x=0, y=1):
        if isinstance(text, bool):
            if text:
                # Label each point with its sample id (row id)
                pass
        for hue, hue_df in self.reduced_data.groupby(self.hue):
            for symbol, symbol_df in hue_df.groupby(self.symbol):
                self.markerplotter(symbol_df[:, x], symbol_df[:, y], text=text,
                                   symbol=symbol, color=hue, ax=ax)

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
