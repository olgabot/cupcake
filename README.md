# Machine learning visualization

[![](https://img.shields.io/travis/olgabot/cupcake.svg)](https://travis-ci.org/olgabot/cupcake)[![](https://img.shields.io/pypi/v/cupcake.svg)](https://pypi.python.org/pypi/cupcake)

Painlessly visualize results of machine learning analyses

* Free software: BSD license

## Features

* TODO

### `scatterplot`

Highly configurable scatterplot, allowing you to specify the hue, marker,
shape, size, alpha, linewidth and edgecolor of a plot in a single command,
similar to `ggplot2` in R.

Instead of this:

```python
tedious matplotlib code
```

You can do this:

```python
import cupcake as cup

simple cupcake code
```

#### Combine with `seaborn`

Combine `cupcake` with the statistical plotting library `seaborn` to create
grids of configurable scatterplots.

```python
import seaborn as sns
import cupcake as cup
sns.set(style='ticks', context='talk')

iris = sns.load_dataset('iris')

g = sns.FacetGrid(iris)
g.map_dataframe(cup.scatterplot, x='sepal_length', y='sepal_width',
                hue='species', alpha='petal_length', size='petal_width')
```


### `smushplot`

For all your dimensionality reduction needs! Given any high-dimensional dataset,
perform dimensionality reduction and plot the result.
