import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter(df, y_column, x_column=None, y_title="y", x_title="x", color_col=None):
    """
    create a scattter plot
    :param df: numpy 2D array
    :param y_column: y col of the dataframe to be used for y value
    :param x_column:  x col of the dataframe to be used for x value. If None, will x be a np.arange(len(nb of sample))
    :param x_title: x axis title
    :param y_title: y axis title
    :param color_col: column to be used for color
    """
    x = np.arange(df.shape[0]) if x_column is None else df[:, x_column]
    if color_col is not None:
        color_scale = np.unique(df[:, color_col])
        colors = np.linspace(0, 1, len(color_scale))
        color_dict = dict(zip(color_scale, colors))
        color_func = np.vectorize(lambda val: color_dict[val])
        color_df = color_func(df[:, color_col])
        plt.scatter(x=x, y=df[:, y_column], c=color_df)
    else:
        plt.scatter(x=x, y=df[:, y_column])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


def pair_plot(df, header, hue_col=None):
    pddf = pd.DataFrame(df, columns=header)
    hue = pddf.columns[hue_col]if hue_col is not None else None
    sns.pairplot(pddf, hue=hue)
    plt.show()

