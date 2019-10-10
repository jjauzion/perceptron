import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from . import Model
from .. import toolbox


class LogReg(Model.Classification):

    def __init__(self, nb_itertion=1000, learning_rate=0.1, regularization_rate=0, nb_output_unit=1, model_name=None):
        Model.Classification.__init__(self, nb_itertion, learning_rate, regularization_rate, nb_output_unit, model_name)

    def describe(self):
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

    def _compute_hypothesis(self, X):
        """

        :param weight: n by nb_output_unit matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :return: m by nb_output_unit matrix, with m nb of sample
        """
        return self._sigmoid(np.matmul(X, self.weight))

    def _compute_cost(self, X, Y, H):
        """

        self.weight : n by nb_output_unit matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_output_unit matrix, with m nb of sample
        :param H: m by nb_output_unit matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return:
        """
        cost = -1 / Y.shape[0] * (np.sum(np.row_stack(Y) * np.log(H) + (1 - np.row_stack(Y)) * np.log(1 - H)))
        regul = self.regularization / (2 * X.shape[0]) * (np.sum(self.weight ** 2))
        return cost + regul

    def _update_weight(self, X, Y, H):
        """

        self.weight : n by nb_output_unit matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_output_unit matrix, with m nb of sample
        :param H: m by nb_output_unit matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return: n by nb_output_unit matrix
        """
        m = X.shape[0]
        return self.weight - self.learning_rate * (np.matmul(X.T, H - Y) / m + self.regularization * self.weight / m)

    def _compute_accuracy(self, y, y_pred):
        for i in range(y.shape[0]):
            self.confusion_matrix[int(y_pred[i]), int(y[i])] += 1
        total_predicted = np.sum(self.confusion_matrix, axis=1)
        total_true = np.sum(self.confusion_matrix, axis=0)
        true_positive = np.diagonal(self.confusion_matrix)
        self.precision = true_positive / total_true
        self.precision = np.insert(self.precision, 0, np.average(self.precision))
        self.recall = np.zeros(true_positive.shape, dtype=float)
        np.divide(true_positive, total_predicted, out=self.recall, where=(total_predicted != 0))
        self.recall = np.insert(self.recall, 0, np.average(self.recall))
        self.f1score = np.zeros(self.precision.shape, dtype=float)
        tmp = self.precision + self.recall
        np.divide((2 * self.precision * self.recall), tmp, out=self.f1score, where=(tmp != 0))
        self.f1score = np.insert(self.f1score, 0, np.average(self.f1score))
        self.accuracy = np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]

    def fit(self, X, y, verbose=1):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param y: vector of shape (n_samples)
        :param verbose: verbosity level -> 0: nothing is printed ; 1: minimal printing ; 2: plot and print
        :return: y_pred from X after training, vector of shape (n_samples)
        """
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        Y = toolbox.one_hot_encode(y, self.nb_output) if self.nb_output > 1 else y.reshape(-1, 1)
        self.weight = np.random.random((X.shape[1], self.nb_output))
        for i in range(self.nb_iter):
            H = self._compute_hypothesis(X)
            self.weight = self._update_weight(X, Y, H)
            self.cost_history[i] = self._compute_cost(X, Y, H)
        Y_pred = self._compute_hypothesis(X)
        y_pred = self._to_class_id(Y_pred)
        self._compute_accuracy(y, y_pred)
        if verbose >= 1:
            print("Training completed!")
            self.print_accuracy()
        if verbose >= 2:
            self.plot_training()
        return y_pred

    def predict(self, X, verbose=1):
        """
        Make prediction based on x
        :param X: matrix of shape (n_samples, n_feature)
        :param verbose: verbosity level -> 0: nothing is printed ; 1: minimal printing
        :return: ((n_feature x nb_of_class matrix), (n_feature x 1) matrix)
        """
        if self.weight is None:
            self.weight = np.zeros((X.shape[1], 1))
            print("Warning: it seems the model is not yet trained...")
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        if X.shape[1] != self.weight.shape[0]:
            raise ValueError("The input X matrix dimension ({}) doesn't match with model weight shape ({})"
                             .format(X.shape, self.weight.shape))
        y_pred = self._compute_hypothesis(X)
        if verbose >= 1:
            print("Prediction completed!".format())
        if verbose >= 2:
            print(y_pred)
        return self._to_class_id(y_pred), y_pred

    def save_model(self, file):
        with Path(file).open(mode='wb') as fd:
            pickle.dump(self.__dict__, fd)
