
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


class Test__ReducedPlotter(object):
    nrow = 10
    ncol = 20

    vector = np.random.negative_binomial(n=1000, p=0.2, size=nrow * ncol)
    matrix = vector.reshape(nrow, ncol)

    symbol_kws = dict(marker='o', marker_order=None, text=False,
                      text_order=None, linewidth=1, linewidth_order=None,
                      edgecolor='k', edgecolor_order=None)

    def test_establish_variables(self):
        from cupcake.reduction import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)

        assert isinstance(p.high_dimensional_data, pd.DataFrame)
        pdt.assert_frame_equal(p.high_dimensional_data,
                               pd.DataFrame(self.matrix))

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
        matrix = self.vector.reshape((1, self.nrow, self.ncol))
        p.establish_variables(matrix)

    def test_establish_symbols_defaults(self):
        from cupcake.reduction import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**self.symbol_kws)

        pdt.assert_series_equal(p.symbol, pd.Series(['o']*self.nrow))
        pdt.assert_series_equal(p.linewidth, pd.Series([1]*self.nrow))
        pdt.assert_series_equal(p.edgecolor, pd.Series(['k']*self.nrow))
        assert p.text == False

    def test_establish_symbols_text_true(self):
        from cupcake.reduction import _ReducedPlotter

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = True

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**symbol_kws)

        pdt.assert_series_equal(p.symbol,
                                pd.Series(map(str, list(range(self.nrow)))))
        assert p.text

    def test_establish_symbols_text_series(self):
        from cupcake.reduction import _ReducedPlotter

        half = int(self.nrow/2.)
        text = pd.Series((['A'] * half) + (['B'] * half))

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = text

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**symbol_kws)

        symbol = pd.Categorical(text, ordered=True, categories=['A', 'B'])
        pdt.assert_categorical_equal(p.symbol, symbol)
        assert p.text

    def test_establish_symbols_text_series_not_str(self):
        from cupcake.reduction import _ReducedPlotter

        half = half = int(self.nrow/2.)
        text = pd.Series(([1] * half) + ([2] * half)).map(str)

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = text

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**symbol_kws)

        symbol = pd.Categorical(text, ordered=True,
                                categories=['1', '2'])
        pdt.assert_categorical_equal(p.symbol, symbol)
        assert p.text



