from pathlib import Path
import numpy as np

from src import dataframe
from src import processing
from src import toolbox


def check_test(df_tool, df=None, df_file=None, model=None, model_file=None):
    if bool(df) == bool(df_file):
        raise AttributeError("Only one of dataframe and df_file can be defined")
    if bool(model) == bool(model_file):
        raise AttributeError("Only one of model and model_file can be defined")
    if df is None:
        df = dataframe.DataFrame(import_scale_and_label=df_tool)
        try:
            df.read_from_csv(df_file, header=False)
        except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
            print("Could not read file '{}' because : {}".format(Path(df_file), err))
            exit(0)
        df.scale(exclude_col=1)
    X = np.delete(df.data, 1, axis=1)
    y = df.data[:, 1]
    if model is None:
        model = processing.NeuralNetwork()
        try:
            model.load_model(model_file)
        except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
            print("Could not read file '{}' because : {}".format(Path(model_file), err))
            exit(0)
    y_pred, _ = model.predict(X)
    print("y_pred =\n{}\ny_pred_val =\n{}".format(y_pred, _))
    confusion_matrix, precision, recall, f1score, accuracy = toolbox.compute_accuracy(y, y_pred)
    toolbox.print_accuracy(precision, recall, f1score, accuracy, 2, confusion_matrix=confusion_matrix)


if __name__ == "__main__":
    check_test("model/data_train_scale.pkl", df_file="data/data_test.csv", model_file="model/m1.pkl")
