import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from .. import toolbox


class Classification:

    def __init__(self, nb_iteration=1000, learning_rate=0.1, regularization_rate=0, nb_output_unit=1, model_name=None):
        """

        :param nb_iteration:
        :param learning_rate:
        :param regularization_rate:
        :param model_name:
        """
        self.nb_iter = nb_iteration
        self.learning_rate = learning_rate
        self.regularization = 0 if regularization_rate is None else regularization_rate
        self.name = model_name
        self.nb_output = nb_output_unit
        nb_class = self.nb_output if self.nb_output > 1 else 2
        self.confusion_matrix = np.zeros((nb_class, nb_class), dtype=int)
        self.precision = [-1]
        self.recall = [-1]
        self.f1score = [-1]
        self.accuracy = -1
        self.cost_history = np.zeros(nb_iteration)
        self.weight = None

    def describe(self):
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("\nPerformance:")
        self.print_accuracy()

    def print_accuracy(self, class_name=None):
        """
        Print model's performance scores (accuracy, recall, precision, F1score)
        :param class_name: list containing the name of each class in order
        """
        nb_class = self.nb_output if self.nb_output > 1 else 2
        class_name = class_name if class_name is not None else [str(elm) for elm in range(nb_class)]
        class_name = ["Average"] + class_name
        col_padding = [15] + [max(9, len(elm)) for elm in class_name]
        line = [
            "".ljust(col_padding[0], " "),
            "Precision".ljust(col_padding[0], " "),
            "Recall".ljust(col_padding[0], " "),
            "F1score".ljust(col_padding[0], " ")
        ]
        for i in range(len(class_name)):
            line[0] += class_name[i].ljust(col_padding[i + 1], " ")
            line[1] += "{}%".format(str(round(self.precision[i] * 100, 2))).ljust(col_padding[i + 1], " ")
            line[2] += "{}%".format(str(round(self.recall[i] * 100, 2))).ljust(col_padding[i + 1], " ")
            line[3] += "{}%".format(str(round(self.f1score[i] * 100, 2))).ljust(col_padding[i + 1], " ")
        print("\n".join(line))
        print("{title:<{width1}}{val:<{width2}}".format(
            title="Accuracy", width1=col_padding[0], val=str(round(self.accuracy * 100, 2)) + "%", width2=col_padding[1]))

    def plot_training(self):
        """Plot training curve convergence"""
        fig = plt.figure("Training convergence")
        plt.plot(self.cost_history)
        plt.title("Cost history")
        plt.xlabel("nb of iterations")
        plt.ylabel("Cost")
        plt.show()

    def load_model(self, file):
        """load an existing model from a pickle file"""
        with Path(file).open(mode='rb') as fd:
            try:
                model = pickle.load(fd)
            except (pickle.UnpicklingError, EOFError) as err:
                raise ValueError("Can't load model from '{}' because : {}".format(file, err))
        if not isinstance(model, dict):
            raise ValueError("Given file '{}' is not a valid model".format(file))
        for key in model.keys():
            if key not in self.__dict__.keys():
                raise ValueError("Given file '{}' is not a valid model".format(file))
        self.__dict__.update(model)
        return True

    def save_model(self, file):
        with Path(file).open(mode='wb') as fd:
            pickle.dump(self.__dict__, fd)

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def _to_class_id(self, Y_pred):
        """
        Converts a one-hot encoded matrix (where each column is a class and each row a sample, and
        where the value correspond to the probability that sample i is of class j),
        to a vector with the most probable class ID for each sample.
        Ex: Y_pred=[[0.9, 0.1, 0.2], [0.1, 0.1, 0.9]] will return [0, 3]
        :param Y_pred: m by nb_output_unit matrix, with m is the nb of sample
        :return: vector of size m (where m is the number of sample) containing the predicted class number for each sample
        """
        if self.nb_output > 1:
            return Y_pred.argmax(axis=1)
        else:
            return np.round(Y_pred).flatten()
