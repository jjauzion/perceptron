import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from . import Model
from .. import toolbox


class LogReg(Model.Classification):

    def __init__(self, learning_rate=0.1, regularization_rate=0, nb_output_unit=1, model_name=None):
        Model.Classification.__init__(self, learning_rate, regularization_rate, nb_output_unit, model_name)

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

    def fit(self, X, y, nb_iteration=None, verbose=1):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param y: vector of shape (n_samples)
        :param nb_iteration: number of iteration to run. If None, will take number stored in self.nb_iteration
        :param verbose: verbosity level -> 0: nothing is printed ; 1: minimal printing ; 2: plot and print
        :return: y_pred from X after training, vector of shape (n_samples)
        """
        if nb_iteration is not None:
            self.nb_iter = nb_iteration
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        Y = toolbox.one_hot_encode(y, self.nb_output) if self.nb_output > 1 else y.reshape(-1, 1)
        self.weight = np.random.random((X.shape[1], self.nb_output))
        for i in range(self.nb_iter):
            H = self._compute_hypothesis(X)
            self.weight = self._update_weight(X, Y, H)
            self.cost_history.append(self._compute_cost(X, Y, H))
        self.nb_iteration_ran += i + 1
        Y_pred = self._compute_hypothesis(X)
        y_pred = self._to_class_id(Y_pred)
        self.compute_accuracy(y, y_pred)
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
