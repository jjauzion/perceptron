import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


class LogReg:

    def __init__(self, nb_itertion=1000, learning_rate=0.1, nb_class=1, regularization_rate=0, model_name=None):
        self.nb_iter = nb_itertion
        self.learning_rate = learning_rate
        self.nb_class = nb_class
        self.regularization = 0 if regularization_rate is None else regularization_rate
        self.name = model_name
        self.confusion_matrix = np.zeros((nb_class, nb_class), dtype=int)
        self.precision = [-1]
        self.recall = [-1]
        self.f1score = [-1]
        self.accuracy = -1
        self.weight = None
        self.cost_history = np.zeros((nb_itertion, nb_class))

    def describe(self):
        print("Model: {}".format(self.name))
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

    def print_accuracy(self, class_name=None):
        """

        :param class_name: list containing the name of each class in order
        """
        class_name = class_name if class_name is not None else [str(elm) for elm in range(self.nb_class)]
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

    def plot_training(self, class_name=None):
        fig = plt.figure("Training convergence")
        for i in range(self.nb_class):
            plt.plot(self.cost_history[:, i])
        plt.legend(list(range(self.nb_class)))
        plt.title("Cost history")
        plt.xlabel("nb of iterations")
        plt.ylabel("Cost")
        plt.show()

    def load_model(self, file):
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

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _to_class_id(Y_pred):
        """

        :param Y_pred: m by nb_class matrix, with m nb of sample
        :return: m by 1 matrix -> predicted class number
        """
        return Y_pred.argmax(axis=1)

    def _compute_hypothesis(self, X):
        """

        :param weight: n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :return: m by nb_class matrix, with m nb of sample
        """
        return self._sigmoid(np.matmul(X, self.weight))

    def _compute_cost(self, X, Y, H):
        """

        self.weight : n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_class matrix, with m nb of sample
        :param H: m by nb_class matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return:
        """
        cost = -1 / X.shape[0] * (np.matmul(Y.T, np.log(H)) + np.matmul((1 - Y).T, np.log(1 - H)))
        regul = self.regularization / (2 * X.shape[0]) * (np.matmul(self.weight.T, self.weight))
        return np.diagonal(cost + regul)

    def _update_weight(self, X, Y, H):
        """

        self.weight : n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_class matrix, with m nb of sample
        :param H: m by nb_class matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return: n by nb_class matrix
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
        self.recall = true_positive / total_predicted
        self.recall = np.insert(self.recall, 0, np.average(self.recall))
        self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.f1score = np.insert(self.f1score, 0, np.average(self.f1score))
        self.accuracy = np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]

    @staticmethod
    def _get_multi_class_y(y, nb_class):
        def is_class(val, class_nb):
            return 1 if val == class_nb else 0
        Y = np.ones((y.shape[0], nb_class), dtype="float64")
        for i in range(nb_class):
            Y[:, i] = [is_class(val, i) for val in y]
        return Y

    def fit(self, X, y, verbose=1):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param y: vector of shape (n_samples)
        :param verbose: verbosity level -> 0: nothing is printed ; 1: minimal printing ; 2: plot and print
        :return: y_pred from X after training, vector of shape (n_samples)
        """
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        Y = self._get_multi_class_y(y, self.nb_class) if self.nb_class > 1 else y.reshape(-1, 1)
        self.weight = np.random.random((X.shape[1], self.nb_class))
        for i in range(self.nb_iter):
            H = self._compute_hypothesis(X)
            self.weight = self._update_weight(X, Y, H)
            self.cost_history[i, :] = self._compute_cost(X, Y, H)
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
        :return: ((n_feature x nb_of_class matrx), (n_feature x 1) matrix)
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
