import matplotlib.pyplot as plt
import pytest
import seaborn as sns

class TestPlotterMixin(object):
    pass


class TestScatterPlotter(object):
    pass

    @pytest.fixture
    def iris(self):
        return sns.load_dataset('iris')

    def test_scatterplots(self, iris):
        """Smoke test of high level scatterplot options"""

        from cupcake.scatter import scatterplot

        scatterplot(iris)
        plt.close('all')

        scatterplot(iris, text=True)
        plt.close('all')

        scatterplot(iris, hue='species')
        plt.close('all')

        scatterplot(iris, linewidth='species')
        plt.close('all')

        scatterplot(iris, edgecolor='species')
        plt.close('all')
