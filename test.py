import argparse
from pathlib import Path
import numpy as np

from src import processing
from src import dataframe


parser = argparse.ArgumentParser()
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
parser.add_argument("-t", "--topology", type=int, nargs="+", default=[31, 5, 1], help="topology of the NN, ex: -t 4 4 1")
parser.add_argument("-gc", "--grad_checking", action="store_true", help="topology of the NN, ex: -t 4 4 1")
args = parser.parse_args()
data = np.array([
    [10, 1, 20],
    [30, 1, 20],
    [1, 1, 2],
    [15, 1, 10],
    [-40, 0, -20],
    [-50, 0, -30],
    [-1, 0, -2],
    [-12, 0, -20]
])
df = dataframe.DataFrame(data=data)
df.scale(exclude_col=1)
df.describe()
model = processing.NeuralNetwork(topology=args.topology, nb_iteration=1000, regularization_rate=0)
model.fit(np.delete(df.data, 1, axis=1), df.data[:, 1], seed=4, gradient_checking=args.grad_checking, verbose=2)
y_id, y_pred = model.predict(X=np.array([[20, 20], [-1, -10], [-0.1, -2], [1, 2]]))
print("model1\n", y_pred)
model = processing.LogReg(nb_output_unit=1)
model.fit(np.delete(df.data, 1, axis=1), df.data[:, 1], verbose=2)
