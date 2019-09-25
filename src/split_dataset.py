import pandas as pd
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to the dataset to split")
parser.add_argument("--train_output", default="data/data_train.csv", help="file where to save train dataset")
parser.add_argument("--cv_output", default="data/data_test.csv", help="file where to save cross validation dataset")
args = parser.parse_args()

try:
    df = pd.read_csv(args.dataset, index_col=0)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
mask = np.random.rand(df.shape[0]) < 0.8
df_train = df.iloc[mask, :]
df_test = df.iloc[~mask, :]
try:
    df_train.to_csv(args.train_output)
    df_test.to_csv(args.cv_output)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
print("train dataset written to '{}'\ncross validation dataset written to : '{}'".format(args.train_output, args.cv_output))

