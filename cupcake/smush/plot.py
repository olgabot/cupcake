"""
User-facing interface for plotting all dimensionality reduction algorithms
"""

def smushplot(data, smusher='PCA', x=1, y=2, n_components=2, marker='o',
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
    smusher : str or object
        Either a string specifying a valid dimensionality reduction algorithm
        in ``sklearn.decomposition`` or ``sklearn.manifold``, or any object
        with ``fit_transform()`` methods.


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
