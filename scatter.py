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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
    parser.add_argument("-a", "--all", action="store_true", help="Scatter plot of all column in the dataset")
    parser.add_argument("-cx", "--columnX", type=is_positive_int, help="column index to plot on x axis")
    parser.add_argument("-cy", "--columnY", type=is_positive_int, help="column index to plot on y axis")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        df.read_from_csv(args.file, header=True, converts={1: ["B", "M"]})
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    if len(df.data) == 0:
        print("No data in file '{}' :(".format(Path(args.file)))
        exit(0)
    if args.all:
        scatter_all(df)
    else:
        df.scatter(args.columnX, args.columnY, color_col=1)
