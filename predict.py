import argparse
from pathlib import Path
import numpy as np

from src import check_arg
from src import wrapper_fct


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="model file to be used for prediction, pickle format")
parser.add_argument("-dt", "--df_tool", type=str, default="model/data_train_scale.pkl", help="pickle file with df scale and label")
parser.add_argument("-f", "--df_file", type=str, default="data/data_test.csv", help="test dataset file, csv format")
parser.add_argument("-o", "--output", type=str, default="output.csv", help="output file for the predictions")
parser.add_argument("-tcol", "--target_col", type=check_arg.is_positive_int, default=1, help="Index of the column containing the target for prediction")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
args = parser.parse_args()
df_test = wrapper_fct.create_dataframe(args.df_file, header=not args.no_header, scale=Path(args.df_tool))
if args.target_col >= df_test.data.shape[1]:
    print("ERROR: target column index is out of bound")
    exit(0)
df_test.scale(exclude_col=args.target_col)
X = np.delete(df_test.data, args.target_col, axis=1)
y = df_test.data[:, args.target_col]
model = wrapper_fct.load_model(args.model)
y_pred, _ = model.predict(X, verbose=0)
try:
    np.savetxt(args.output, y_pred, delimiter=',', fmt="%i")
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not write file '{}' because : {}".format(Path(args.model), err))
    exit(0)
print("Prediction save to file '{}'".format(Path(args.output)))
