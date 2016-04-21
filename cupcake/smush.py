"""
User-facing interface to all dimensionality reduction algorithms
"""

def smushplot(data, smusher):
    if isinstance(smusher, str):
        # Need to get appropriate smusher from sklearn given the string
        pass
    else:
        # Assume this is already an initialized sklearn object with the
        # ``fit_transform()`` method
        pass
