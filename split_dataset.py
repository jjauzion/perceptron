import pandas as pd
from pathlib import Path
import argparse
import numpy as np

from src import check_arg


train_output = "data/data_train.csv"
test_output = "data/data_test.csv"
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to the dataset to split")
parser.add_argument("--train_output", default=train_output, help="file where to save train dataset. Default = '{}'".format(train_output))
parser.add_argument("--cv_output", default=test_output, help="file where to save cross validation dataset. Default = '{}'".format(test_output))
parser.add_argument("-r", "--ts_ratio", type=check_arg.is_percentage, default=0.8, help="ration of data to put in the training set (eg: 0.8 for 80%%)")
parser.add_argument("-nh", "--no_header", action="store_true", help="Dataset has no header on the first line")
args = parser.parse_args()

try:
    df = pd.read_csv(args.dataset, index_col=0, header=None if args.no_header else 0)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
np.random.seed(10)
mask = np.random.rand(df.shape[0]) < args.ts_ratio
df_train = df.iloc[mask, :]
df_test = df.iloc[~mask, :]
try:
    df_train.to_csv(args.train_output)
    df_test.to_csv(args.cv_output)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
print("train dataset written to '{}'. Size: {}".format(args.train_output, df_train.shape))
print("cross validation dataset written to : '{}'. Size: {}".format(args.cv_output, df_test.shape))

