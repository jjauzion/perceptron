import argparse
from pathlib import Path
import numpy as np

from src import processing
from src import wrapper_fct


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="model file to be used for prediction, pickle format")
parser.add_argument("-dt", "--df_tool", type=str, default="model/data_train_scale.pkl", help="pickle file with df scale and label")
parser.add_argument("-f", "--df_file", type=str, default="data/data_test.csv", help="test dataset file, csv format")
parser.add_argument("-o", "--output", type=str, default="output.csv", help="output file for the predictions")
args = parser.parse_args()
df_test = wrapper_fct.create_dataframe(args.df_file, header=None, scale=Path(args.df_tool))
df_test.scale(exclude_col=1)
X = np.delete(df_test.data, 1, axis=1)
y = df_test.data[:, 1]
model = wrapper_fct.load_model(args.model)
y_pred, _ = model.predict(X, verbose=0)
try:
    np.savetxt(args.output, y_pred, delimiter=',')
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not write file '{}' because : {}".format(Path(args.model), err))
    exit(0)
print("Prediction save to file '{}'".format(Path(args.output)))
