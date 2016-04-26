# Machine learning visualization

[![](https://img.shields.io/travis/olgabot/cupcake.svg)](https://travis-ci.org/olgabot/cupcake)[![](https://img.shields.io/pypi/v/cupcake.svg)](https://pypi.python.org/pypi/cupcake)

Painlessly visualize results of machine learning analyses

* Free software: BSD license

## Features

* TODO

### `smushplot`

For all your dimensionality reduction needs! Given any high-dimensional dataset,
perform dimensionality reduction and plot the result.

#### Colors

Specifying colors for plotting can be kind of confusing, so here is a table of all the possible inputs and outputs with examples.

| #  | `color` | `palette` | `hue`  | `hue_order` | *Output* |
| -- | ------- | --------- | ------ | ----------- | -------- |
| 1. | `None`  | `None`    | `None` | `None`      | One color, auto-assigned (from `ax.color_cycle()`) |
| 2. | `None`  | `None`    | `None` | `list` object | Cannot interpret, raises `ValueError` |
| 3. | `None`  | `None`    | `Series` object | `None`      | Color by group mapping specified in `hue` with **auto-assigned** colors |
| 4. | `None`  | `None`    | `Series` object | `list` object | Color by group mapping specified in `hue` with **auto-assigned** colors, ordered by `hue_order` |
| 5. | `None`  | [valid matplotlib palette name](http://matplotlib.org/examples/color/colormaps_reference.html) e.g."PRGn" or [seaborn palette name](https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html) or list of colors | `None` | `None`      | Every sample a different color, auto-assigned by the row order of the input data |
| 6. | `None`  | [valid matplotlib palette name](http://matplotlib.org/examples/color/colormaps_reference.html) e.g."PRGn" or [seaborn palette name](https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html) or list of colors | `None` | `list` object | If `len(hue_order)=len(data.index)`, then every sample a different color, in the order specified by `hue_order`. Otherwise raises `ValueError` |
| 7. | `None`  | [valid matplotlib palette name](http://matplotlib.org/examples/color/colormaps_reference.html) e.g."PRGn" or [seaborn palette name](https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html) or list of colors | sample id (row name) to group mapping object (`pandas.Series` or `dict`) | None | Group by colors with this palette, groups in alphanumeric order |
| 8. | `None`  | [valid matplotlib palette name](http://matplotlib.org/examples/color/colormaps_reference.html) e.g."PRGn" or [seaborn palette name](https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html) or list of colors | sample id (row name) to group mapping object (`pandas.Series` or `dict`) | `list` object | Group by colors with this palette, in order specified by `hue_order` |




