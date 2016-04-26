"""
User-facing interface for plotting all dimensionality reduction algorithms
"""

def smushplot(data, smusher='pca', n_components=2, marker='o', marker_order=None,
              text=False, text_order=None, linewidth=1, linewidth_order=None,
              edgecolor='k', edgecolor_order=None, smusher_kws=None,
              plot_kws=None):
    if isinstance(smusher, str):
        # Need to get appropriate smusher from sklearn given the string
        pass
    else:
        # Assume this is already an initialized sklearn object with the
        # ``fit_transform()`` method
        pass
