import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src import dataframe


def is_positive_int(value):
    i_value = int(value)
    if i_value < 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive int value.".format(value))
    return i_value


def scatter_all(df):
    fig = plt.figure("scatter")
    plt.subplot()
    nb_col = df.data.shape[1]
    for x in range(nb_col):
        for y in range(x + 1, nb_col):
            plt.scatter(x=df.data[:, x], y=df.data[:, y])
            plt.xlabel(df.header[x])
            plt.ylabel(df.header[y])
            plt.subplot(nb_col, nb_col, y + x * nb_col)
    plt.show()


def scatter(df, x_column, y_column, color_col=None):
    if color_col is not None:
        color_scale = np.unique(df.data[:, color_col])
        colors = np.linspace(0, 1, len(color_scale))
        color_dict = dict(zip(color_scale, colors))
        color_func = np.vectorize(lambda x: color_dict[x])
        color_df = color_func(df.data[:, color_col])
        plt.scatter(x=df.data[:, x_column], y=df.data[:, y_column], c=color_df)
    else:
        plt.scatter(x=df.data[:, x_column], y=df.data[:, y_column])
    plt.xlabel(df.header[x_column])
    plt.ylabel(df.header[y_column])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
    parser.add_argument("-a", "--all", action="store_true", help="Scatter plot of all column in the dataset")
    parser.add_argument("-cx", "--columnX", type=is_positive_int, default=4, help="column index to plot on x axis (0 by default)")
    parser.add_argument("-cy", "--columnY", type=is_positive_int, default=6, help="column index to plot on y axis (2 by default)")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
        df.read_from_csv(args.file, header=True, converts={1: classes, 5: ["Left", "Right"]})
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    if len(df.data) == 0:
        print("No data in file '{}' :(".format(Path(args.file)))
        exit(0)
    df.drop_nan_column()
    df.scale(scale_type="minmax", first_col=3)
    if args.all:
        scatter_all(df)
    else:
        scatter(df, args.columnX, args.columnY, color_col=1)
