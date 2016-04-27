import string

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest
import seaborn as sns
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

    half = int(nrow / 2.)
    groupby = pd.Series((['B'] * half) + (['A'] * half))
    order = ['A', 'B']

    palette = 'PRGn'
    color = 'DarkTeal'

    symbol_kws = dict(marker='o', marker_order=None, text=False,
                          text_order=None, linewidth=1, linewidth_order=None,
                          edgecolor='k', edgecolor_order=None)
    color_kws = dict(color=None, palette=None, hue=None, hue_order=None,
                     saturation=None)

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

    # --- Test figuring out what to reduce --- #
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

    # --- Test internal series making function --- #
    def test__maybe_make_grouper_single_groupby(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)

        test_grouper = p._maybe_make_grouper('o', None, str)
        true_grouper = pd.Series(['o']*self.matrix.shape[0],
                                 index=self.data.index)
        pdt.assert_series_equal(test_grouper, true_grouper)

    def test__maybe_make_grouper_multiple_groupbys(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)

        half = int(self.nrow/2.)
        groupby = pd.Series((['B'] * half) + (['A'] * half))
        order = ['A', 'B']

        test_grouper = p._maybe_make_grouper(groupby, order, str)
        true_grouper = pd.Series(pd.Categorical(groupby, categories=order,
                                      ordered=True), index=self.data.index)
        pdt.assert_series_equal(test_grouper, true_grouper)

    # --- Test assigning plotting symbols --- #
    def test_establish_symbols_defaults(self):
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_symbols(**self.symbol_kws)

        pdt.assert_series_equal(p.symbol, pd.Series(['o']*self.nrow,
                                                    index=self.data.index))
        pdt.assert_series_equal(p.linewidth, pd.Series([1]*self.nrow,
                                                       index=self.data.index))
        pdt.assert_series_equal(p.edgecolor, pd.Series(['k']*self.nrow,
                                                       index=self.data.index))
        assert not p.text

    def test_establish_symbols_text_true(self):
        from cupcake.smush.base import _ReducedPlotter

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = True

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_symbols(**symbol_kws)

        pdt.assert_series_equal(p.symbol,
                                pd.Series(map(str, list(range(self.nrow))),
                                          index=self.data.index))
        assert p.text

    def test_establish_symbols_text_series(self):
        from cupcake.smush.base import _ReducedPlotter

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = self.groupby

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_symbols(**symbol_kws)

        order = sns.utils.categorical_order(self.groupby)
        symbol = pd.Series(pd.Categorical(self.groupby, categories=order,
                                          ordered=True),
                           index=self.data.index)
        pdt.assert_series_equal(p.symbol, symbol)
        assert p.text

    def test_establish_symbols_text_series_ordered(self):
        from cupcake.smush.base import _ReducedPlotter

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = self.groupby
        symbol_kws['text_order'] = self.order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_symbols(**symbol_kws)

        symbol = pd.Series(pd.Categorical(self.groupby, ordered=True,
                                categories=self.order), index=self.data.index)
        pdt.assert_series_equal(p.symbol, symbol)
        assert p.text

    def test_establish_symbols_text_series_not_str(self):
        from cupcake.smush.base import _ReducedPlotter

        half = int(self.nrow/2.)
        text = pd.Series(([1] * half) + ([2] * half)).map(str)

        symbol_kws = self.symbol_kws.copy()
        symbol_kws['text'] = text

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_symbols(**symbol_kws)

        symbol = pd.Series(pd.Categorical(text, ordered=True,
                                categories=['1', '2']), index=self.data.index)
        pdt.assert_series_equal(p.symbol, symbol)
        assert p.text

    # --- Test assigning colors --- #
    def test_establish_colors_all_none(self):
        # Option 1. All parameters are set to default values
        from cupcake.smush.base import _ReducedPlotter

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**self.color_kws)

        assert p.n_colors == 1
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    @pytest.mark.xfail
    def test_establish_colors_hue_order(self):
        # Option 2. hue_order is specified but nothing else is
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['hue_order'] = self.order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

    def test_establish_colors_hue(self):
        # Option 3. "hue" is specified but nothing else is
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['hue'] = self.groupby

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 2
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_hue_hue_order(self):
        # Option 4. Both "hue" and "hue_order" are specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['hue'] = self.groupby
        color_kws['hue_order'] = self.order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        test_grouped = p.high_dimensional_data.groupby(p.color)

        assert p.n_colors == 2
        assert len(test_grouped) == p.n_colors

    def establish_colors_palette(self):
        # Option 5. "palette" is specified but nothing else is
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['palette'] = self.palette

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == self.nrow
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_palette_hue_order(self):
        # Option 6a. "palette" and "hue_order" are specified
        from cupcake.smush.base import _ReducedPlotter

        # Reverse the index order
        hue_order = self.data.index[::-1]

        color_kws = self.color_kws.copy()
        color_kws['palette'] = self.palette
        color_kws['hue_order'] = hue_order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == self.nrow
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    @pytest.mark.xfail
    def establish_colors_palette_hue_order_incorrect_length(self):
        # Option 6b. "palette" and "hue_order" are specified, but hue_order
        # isn't correct length
        from cupcake.smush.base import _ReducedPlotter

        # Reverse the index order
        hue_order = self.data.index[::-1]
        hue_order = hue_order[:self.half]

        color_kws = self.color_kws.copy()
        color_kws['palette'] = self.palette
        color_kws['hue_order'] = hue_order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

    def establish_colors_palette_hue(self):
        # Option 7. "palette" and "hue" are specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['palette'] = self.palette
        color_kws['hue'] = self.groupby

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 2
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_palette_hue_hue_order(self):
        # Option 8. "palette", "hue", and "hue_order" are specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['palette'] = self.palette
        color_kws['hue'] = self.groupby
        color_kws['hue_order'] = self.order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 2
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_color(self):
        # Option 9. "color" is specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 1
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_color_hue_order(self):
        # Option 10. "color" and "hue_order" are specified
        from cupcake.smush.base import _ReducedPlotter

        # Reverse the index so hue_order is different from original order
        hue_order = self.data.index[::-1]

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color
        color_kws['hue_order'] = hue_order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == self.nrow
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    @pytest.mark.xfail
    def establish_colors_color_hue_order_incorrect_length(self):
        # Option 10. "color" and "hue_order" are specified, but "hue_order" is
        #  the incorrect length
        from cupcake.smush.base import _ReducedPlotter

        # Reverse the index so hue_order is different from original order
        hue_order = self.data.index[::-1]
        hue_order = hue_order[:self.half]

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color
        color_kws['hue_order'] = hue_order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

    def establish_colors_color_hue(self):
        # Option 11. "color" and "hue" are specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color
        color_kws['hue'] = self.groupby

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 2
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    def establish_colors_color_hue_hue_order(self):
        # Option 12. "color", "hue", and "hue_order" are specified
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color
        color_kws['hue'] = self.groupby
        color_kws['hue_order'] = self.order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)

        assert p.n_colors == 2
        assert len(p.high_dimensional_data.groupby(p.color)) == p.n_colors

    @pytest.fixture(params=[None, 'hue'])
    def hue(self, request):
        if request.param is None:
            return request.param
        else:
            return self.groupby

    @pytest.fixture(params=[None, 'hue_order'])
    def hue_order(self, request):
        if request.param is None:
            return request.param
        else:
            return self.order

    @pytest.mark.xfail
    def establish_colors_color_palette(self, hue, hue_order):
        # Option 13-16. "color", and "palette" are specified but incompatible
        from cupcake.smush.base import _ReducedPlotter

        color_kws = self.color_kws.copy()
        color_kws['color'] = self.color
        color_kws['palette'] = self.palette
        color_kws['hue'] = hue
        color_kws['hue_order'] = hue_order

        p = _ReducedPlotter()
        p.establish_variables(self.data)
        p.establish_colors(**color_kws)
