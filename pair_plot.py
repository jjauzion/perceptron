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
    parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        df = pd.read_csv(args.file, index_col=0)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError, OSError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    pd.options.display.width = 0
    num_df = df.drop(labels=df.columns[[1, 2, 3, 4]], axis=1)
    g = sns.pairplot(num_df.sample(100), hue="Hogwarts House", dropna=True)
    plt.show()
