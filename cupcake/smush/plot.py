"""
User-facing interface for plotting all dimensionality reduction algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import pandas as pd

from ..scatter import PlotterMixin


class SmushPlotter(PlotterMixin):
    """Base class for dimensionality reduction plots

    Must be mixed with something that creates:
    - ``self.smusher`` object
    - ``self.high_dimensional_data`` dataframe

    """

    def establish_data(self):
        """Use reduction algorithm to smush high-dimensional data into a smaller space"""
        matrix = self.smusher.fit_transform(self.high_dimensional_data)
        columns = np.arange(matrix.shape[1], dtype=int) + 1
        self.plot_data = pd.DataFrame(matrix, columns=columns,
                                      index=self.high_dimensional_data.index)

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

        pass


def smushplot(data, smusher='PCA', x=0, y=1, n_components=2, marker='o',
              marker_order=None, text=False, text_order=None, linewidth=1,
              linewidth_order=None, edgecolor='k', edgecolor_order=None,
              smusher_kws=None, plot_kws=None):
    """Plot high dimensional data in 2d space

    Parameters
    ----------
    data : pandas.DataFrame or numpy.array
        A (n_samples, m_features) wide matrix of observations. The samples
        (rows) will be plotted relative to the reduced representation of the
        features (columns)
    smusher : str or object or None
        Either a string specifying a valid dimensionality reduction algorithm
        in ``sklearn.decomposition`` or ``sklearn.manifold``, or any object
        with ``fit_transform()`` methods. If ``None``, then ``data`` is assumed
        to be already reduced
    x, y : int
        0-based counting of which components to plot as the x- and y-axes. For
        example, to plot component 4 on the x-axis and component 1 on the y,
        do ``x=3, y=0``.
    n_components : int
        Number of components to use when reducing dimensionality

    Notes
    -----


    """
    if isinstance(smusher, str):
        # Need to get appropriate smusher from sklearn given the string
        pass
    else:
        # Assume this is already an initialized sklearn object with the
        # ``fit_transform()`` method
        pass
