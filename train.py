import argparse
from pathlib import Path
import numpy as np

from src import processing
from src import dataframe


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="file to describe, shall be csv format")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
parser.add_argument("-t", "--topology", type=int, nargs="+", default=[31, 5, 1], help="topology of the NN, ex: -t 4 4 1")
parser.add_argument("-gc", "--grad_checking", action="store_true", help="topology of the NN, ex: -t 4 4 1")
parser.add_argument("-i", "--iteration", type=int, help="topology of the NN, ex: -t 4 4 1")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=not args.no_header, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.scale(exclude_col=1)
df.describe()
model = processing.NeuralNetwork(topology=args.topology, nb_itertion=1000, regularization_rate=0)
model.train(np.delete(df.data, 1, axis=1), df.data[:, 1], seed=4, gradient_checking=args.grad_checking, verbose=2)
print("model1\n", y_pred)
