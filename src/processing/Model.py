import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from .. import toolbox


class Classification:

    def __init__(self, learning_rate=0.1, regularization_rate=0, nb_output_unit=1, model_name=None):
        """

        :param nb_iteration:
        :param learning_rate:
        :param regularization_rate:
        :param model_name:
        """
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
        self.cost_history = []
        self.weight = None
        self.nb_iteration_ran = 0

    def describe(self):
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("Trained on {} iterations".format(self.nb_iteration_ran))
        print("\nPerformance:")
        self.print_accuracy()

    def compute_accuracy(self, y, y_pred):
        self.confusion_matrix, self.precision, self.recall, self.f1score, self.accuracy = \
            toolbox.compute_accuracy(y, y_pred)

    def print_accuracy(self, class_name=None):
        """
        Print model's performance scores (accuracy, recall, precision, F1score)
        :param class_name: list containing the name of each class in order
        """
        nb_class = self.nb_output if self.nb_output > 1 else 2
        toolbox.print_accuracy(self.precision, self.recall, self.f1score, self.accuracy, nb_class, class_name)

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
                raise ValueError("Given file '{}' is not a valid model. Unexpected key :'{}'".format(file, key))
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
