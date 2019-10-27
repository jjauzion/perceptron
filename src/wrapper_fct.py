from pathlib import Path
import numpy as np

from src import dataframe
from src import processing
from src import toolbox


def check_test(target_col, df_tool=None, df_file=None, df=None, model=None, model_file=None, verbose=1):
    if bool(df) == bool(df_file):
        raise AttributeError("Only one of dataframe and df_file can be defined")
    if df_file is not None and df_tool is None:
        raise AttributeError("df scale and label is required if df is imported from file")
    if bool(model) == bool(model_file):
        raise AttributeError("Only one of model and model_file can be defined")
    if df is None:
        df = dataframe.DataFrame(import_scale_and_label=df_tool)
        try:
            df.read_from_csv(df_file, header=False)
        except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
            print("Could not read file '{}' because : {}".format(Path(df_file), err))
            exit(0)
        df.scale(exclude_col=target_col)
    X = np.delete(df.data, target_col, axis=1)
    y = df.data[:, target_col]
    y_1hot = toolbox.one_hot_encode(y)
    if model is None:
        model = processing.NeuralNetwork()
        try:
            model.load_model(model_file)
        except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
            print("Could not read file '{}' because : {}".format(Path(model_file), err))
            exit(0)
    y_pred, h = model.predict(X, verbose=0)
    confusion_matrix, precision, recall, f1score, accuracy = toolbox.compute_accuracy(y, y_pred)
    loss = model._compute_cost(y_1hot, h)
    if verbose >= 1:
        toolbox.print_accuracy(precision, recall, f1score, accuracy, 2, confusion_matrix=confusion_matrix)
    return f1score[0], loss


def create_dataframe(file, header, converts=None, scale=None):
    if bool(converts) and bool(scale):
        raise AttributeError("Can't define both scale and converts argument")
    try:
        df = dataframe.DataFrame(import_scale_and_label=scale)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(scale, err))
        exit(0)
    try:
        df.read_from_csv(file, header=header, converts=converts)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(file), err))
        exit(0)
    return df


def load_model(file):
    model = processing.NeuralNetwork()
    try:
        model.load_model(Path(file))
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(file), err))
        exit(0)
    return model
