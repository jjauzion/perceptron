import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src import dataframe


if __name__ == "__main__":
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, default="data/dataset_train.csv", help="dataset file, shall be csv format")
    parser.add_argument("--index_col", type=int, default=None, help="index of the column to used as a index for the dataframe")
    parser.add_argument("--hue_col", type=int, default=None, help="column index to use to color the plot")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        if args.index_col is not None:
            df = pd.read_csv(args.file, index_col=args.index_col)
        else:
            df = pd.read_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError, OSError) as err:
        print("Could not read or parse file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    # num_df = df.drop(labels=df.columns[[1, 2, 3, 4]], axis=1)
    hue_col = df.columns[args.hue_col] if args.hue_col is not None else None
    g = sns.pairplot(df.sample(100), hue=hue_col, dropna=True)
    print("done")
    plt.show()
