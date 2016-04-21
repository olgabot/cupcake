
import numpy as np
import pandas as pd
import pytest


class Test__ReducedPlotter(object):
    nrow = 10
    ncol = 20

    vector = np.random.negative_binomial(n=1000, p=0.2, size=nrow * ncol)
    matrix = vector.reshape(nrow, ncol)

    def test_establish_variables(self):
        from cupcake.reduction import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)

        assert isinstance(p.high_dimensional_data, pd.DataFrame)

    @pytest.mark.xfail(reason='High dimensional data provided is actually '
                              'small')
    def test_establish_variables_too_few_axes(self):
        from cupcake.reduction import _ReducedPlotter

        p = _ReducedPlotter()
        matrix = self.vector.reshape(1, self.nrow * self.ncol)
        p.establish_variables(matrix)

    @pytest.mark.xfail(reason='High dimensional data provided has too many '
                              'axes')
    def test_establish_variables_too_many_axes(self):
        from cupcake.reduction import _ReducedPlotter

        p = _ReducedPlotter()
        matrix = self.vector.reshape(1, self.nrow, self.ncol)
        p.establish_variables(matrix)
