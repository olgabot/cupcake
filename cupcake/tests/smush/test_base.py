import string

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest
from sklearn.decomposition import PCA


class Test__ReducedPlotter(object):
    nrow = 10
    ncol = 20

    vector = np.random.negative_binomial(n=1000, p=0.2, size=nrow * ncol)
    matrix = vector.reshape(nrow, ncol)
    data = pd.DataFrame(matrix, index=list(string.ascii_lowercase[:nrow]),
                        columns=list(string.ascii_uppercase[:ncol]))

    data = pd.DataFrame(matrix.copy())
    data.index.name = 'Samples'
    data.columns.name = 'Features'

    symbol_kws = dict(marker='o', marker_order=None, text=False,
                      text_order=None, linewidth=1, linewidth_order=None,
                      edgecolor='k', edgecolor_order=None)

    def test_establish_reducer_make_new(self):
        from cupcake.smush.base import _ReducedPlotter

        pca_kws = {}
        n_components = 2
        reducer = PCA(n_components=n_components, **pca_kws)

        p = _ReducedPlotter()
        p.establish_reducer(PCA, n_components, {})

        assert isinstance(p.reducer, type(reducer))
        pdt.assert_dict_equal(p.reducer.get_params(), reducer.get_params())

    def test_establish_reducer_use_existing(self):
        from cupcake.smush.base import _ReducedPlotter

        pca_kws = {}
        n_components = 2
        reducer = PCA(n_components=n_components, **pca_kws)

        p = _ReducedPlotter()
        p.establish_reducer(reducer)

        assert isinstance(p.reducer, type(reducer))
        pdt.assert_dict_equal(p.reducer.get_params(), reducer.get_params())

    def test_establish_variables_matrix(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)

        assert isinstance(p.high_dimensional_data, pd.DataFrame)
        pdt.assert_frame_equal(p.high_dimensional_data,
                               pd.DataFrame(self.matrix))
        assert p.group_label is None
        assert p.value_label is None

    def test_establish_variables_dataframe_named_axes(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)

        pdt.assert_frame_equal(p.high_dimensional_data, self.data)
        assert p.group_label == 'Samples'
        assert p.value_label == "Features"

    @pytest.mark.xfail(reason='High dimensional data provided is too small')
    def test_establish_variables_too_few_axes(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        matrix = self.vector.reshape(1, self.nrow * self.ncol)
        p.establish_variables(matrix)

    @pytest.mark.xfail(reason='High dimensional data provided has too many '
                              'axes')
    def test_establish_variables_too_many_axes(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        matrix = self.vector.reshape((1, self.nrow, self.ncol))
        p.establish_variables(matrix)

    def test__maybe_make_grouper_single_attribute(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)

        test_grouper = p._maybe_make_grouper('o', None, str)
        true_grouper = pd.Series(['o']*self.matrix.shape[0],
                                 index=self.data.index)
        pdt.assert_series_equal(test_grouper, true_grouper)

    def test__maybe_make_grouper_multiple_attributes(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)

        half = int(self.nrow/2.)
        attribute = pd.Series((['B'] * half) + (['A'] * half))
        order = ['A', 'B']

        test_grouper = p._maybe_make_grouper(attribute, order, str)
        true_grouper = pd.Categorical(attribute, categories=order,
                                      ordered=True)
        pdt.assert_categorical_equal(test_grouper, true_grouper)

    def test_establish_symbols_defaults(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**self.symbol_kws)

        pdt.assert_series_equal(p.symbol, pd.Series(['o']*self.nrow))
        pdt.assert_series_equal(p.linewidth, pd.Series([1]*self.nrow))
        pdt.assert_series_equal(p.edgecolor, pd.Series(['k']*self.nrow))
        assert p.text == False

    def test_establish_symbols_text_true(self):
        from cupcake.smush.base import _ReducedPlotter

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = True

        p = _ReducedPlotter()
        p.establish_variables(self.matrix)
        p.establish_symbols(**symbol_kws)

        pdt.assert_series_equal(p.symbol,
                                pd.Series(map(str, list(range(self.nrow)))))
        assert p.text

    def test_establish_symbols_text_series(self):
        from cupcake.smush.base import _ReducedPlotter

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
        from cupcake.smush.base import _ReducedPlotter

        half = int(self.nrow/2.)
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

    def test__maybe_make_grouper(self):
        pass
