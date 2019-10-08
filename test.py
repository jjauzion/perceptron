import argparse
from pathlib import Path
import numpy as np

from src import processing
from src import dataframe


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, help="file to describe, shall be csv format")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
parser.add_argument("-t", "--topology", type=int, nargs="+", default=[31, 5, 1], help="topology of the NN, ex: -t 4 4 1")
args = parser.parse_args()
if args.file is not None:
    df = dataframe.DataFrame()
    try:
        df.read_from_csv(args.file, header=not args.no_header, converts={1: ["B", "M"]})
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
else:
    data = np.array([
        [10, 1, 20],
        [30, 1, 20],
        [1, 1, 2],
        [15, 1, 10],
        [-40, 0, -20],
        [-50, 0, -30],
        [-1, 0, -2],
    ])
    df = dataframe.DataFrame(data=data)
df.scale(exclude_col=1)
df.describe()
model = processing.NeuralNetwork(topology=args.topology, nb_itertion=100, regularization_rate=0)
model.train(np.delete(df.data, 1, axis=1), df.data[:, 1], gradient_checking=True, verbose=2)
y_id, y_pred = model.predict(X=np.array([[20, 20], [-1, -10], [-0.1, -2], [1, 2]]))
print(y_pred)
