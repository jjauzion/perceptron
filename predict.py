import argparse
from pathlib import Path
import numpy as np

from src import processing
import train


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--df_file", type=str, default="data/data_test.csv", help="test dataset file, csv format")
parser.add_argument("-o", "--output", type=str, default="predicition.csv", help="output file for the predictions")
parser.add_argument("model", type=str, help="model file to be used for prediction, pickle format")
parser.add_argument("df_tool", type=str, help="pickle file with df scale and label")
args = parser.parse_args()
df_test = train.create_dataframe(args.df_file, header=None, scale=Path(args.df_tool))
df_test.scale(exclude_col=1)
X = np.delete(df_test.data, 1, axis=1)
y = df_test.data[:, 1]
model = processing.NeuralNetwork()
model.load_model(args.model)
y_pred, _ = model.predict(X, verbose=0)
np.savetxt(args.output, y_pred, delimiter=',')
print("Prediction save to file '{}'".format(Path(args.output)))
