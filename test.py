import argparse
from pathlib import Path
import numpy as np

from src import processing
from src import dataframe


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="file to describe, shall be csv format")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=not args.no_header, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.scale(exclude_col=1)
df.describe()
model = processing.NeuralNetwork(topology=[31, 5, 5, 1], nb_class=2, regularization_rate=0.1)
model.train(np.delete(df.data, 1, axis=1), df.data[:, 1])