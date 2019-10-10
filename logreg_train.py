from pathlib import Path
import argparse

from src import dataframe
from src import processing


def is_positive(value):
    f_value = float(value)
    if f_value < 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive value.".format(value))
    return f_value


def is_positive_int(value):
    i_value = int(value)
    if i_value < 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive int value.".format(value))
    return i_value


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="train dataset file to be used")
parser.add_argument("-n", "--name", type=str, default="model", help="name of the model")
parser.add_argument("-o", "--save_dir", type=str, default="model", help="directory where to save model and df")
parser.add_argument("-s", "--scale", type=str, choices=["minmax", "meannorm"], default="minmax", help="type of scale to be used for data normalization")
parser.add_argument("-r", "--regul_rate", type=is_positive, default=None, help="Value of the regularization rate. 0 or None if no regul (default).")
parser.add_argument("-lr", "--learning_rate", type=is_positive, default=0.1, help="Learning rate value (0.1 by default)")
parser.add_argument("-i", "--nb_iter", type=is_positive_int, default=1000, help="number of iteration (1000 by default)")
parser.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2], help="verbosity level: 0->silent, 1->result print, 2->plot show")
parser.add_argument("-d", "--drop_col_6_7_16", action="store_true", help="drop col 5, 6 and 16 (useless col, see pairplot")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    df.read_from_csv(args.file, header=True, converts={1: classes, 5: ["Left", "Right"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
if not Path(args.save_dir).is_dir():
    print("Directory '{}' does not exist".format(args.save_dir))
    exit(0)
df.drop_column([0])             # index column
if args.drop_col_6_7_16:
    df.drop_column([5, 6, 15])  # useless column (homogenous distri and correlated variable)
df.drop_nan_column()
df.drop_nan_row()
df.scale(scale_type=args.scale, first_col=1)
model = processing.LogReg(nb_itertion=args.nb_iter, learning_rate=args.learning_rate, nb_class=4,
                          regularization_rate=args.regul_rate, model_name=args.name)
y_pred = model.fit(df.data[:, 1:], df.data[:, 0], verbose=args.verbosity)
model_file = Path(args.save_dir) / "{}.pkl".format(args.name)
try:
    model.save_model(model_file)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Can't save model to '{}' because : {}".format(model_file, err))
print("Model saved to '{}'".format(model_file))
df_tool_file = Path(args.save_dir) / "{}_df_tool.pkl".format(args.name)
try:
    df.save_scale_and_label(df_tool_file)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Can't save dataframe scaler and label to '{}' because : {}".format(model_file, err))
print("Dataframe scaler and label saved to '{}'".format(df_tool_file))
