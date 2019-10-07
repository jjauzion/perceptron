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
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

    def print_accuracy(self, class_name=None):
        """
        Print model's performance scores (accuracy, recall, precision, F1score)
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

    def plot_training(self):
        """Plot training curve convergence"""
        fig = plt.figure("Training convergence")
        for i in range(self.nb_class):
            plt.plot(self.cost_history[:, i])
        plt.legend(list(range(self.nb_class)))
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

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _to_class_id(Y_pred):
        """
        Create a m by 1 matrix with m the number of sample. Takes  as an input the Y_pred matrix createdFor each sample, the class id with most class id matrix for each sample the ass
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


# ##################### NEURAL NETWORK ########################


class NeuralNetwork:
    """
    Build a custom Neural Network.
    List of Attributes:
    weight: List of numpy array with the weight for each layers.
            Each element is a matrix of size (S(j+1), S(j) + 1) with S(j) number of neurons in layer j
    unit: List of numpy array with the neuron value for each layer.
          Each element of the list is a vector of size n with n the number of neurons in layer j
    delta: List of numpy array with the delta value for each unit of each layer. The delta is compute for the backpropagation.
           Each element of the list is a vector of size n with n the number of neurons in layer j
    w_delta: List of numpy array with the delta value for each weight of each layer. The delta is computed after the backpropagation.
             Each element of the list is a matrix of same size as the weight matrix of this layer
    """

    def __init__(self, nb_itertion=1000, learning_rate=0.1, nb_class=1, regularization_rate=0, topology=None, model_name=None):
        """

        :param nb_itertion:
        :param learning_rate:
        :param nb_class:
        :param regularization_rate:
        :param topology: list of size nb_of_layer (including input and output layer) with the nb of neuron for each
        layer : [nb_of_n_in_layer_0, nb_of_n_in_layer_1, ...,nb_of_n_in_layer_L]
        :param model_name:
        """
        self.topology = [] if topology is None else topology
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
        self.unit = None
        self.delta = None
        self.w_delta = None
        self.cost_history = np.zeros((nb_itertion, nb_class))

    def describe(self):
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("Topology: {}".format(self.topology))
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

    def print_accuracy(self, class_name=None):
        """
        Print model's performance scores (accuracy, recall, precision, F1score)
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

    def plot_training(self):
        """Plot training curve convergence"""
        fig = plt.figure("Training convergence")
        for i in range(self.nb_class):
            plt.plot(self.cost_history[:, i])
        plt.legend(list(range(self.nb_class)))
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

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _derivative_sigmoid(X):
        return NeuralNetwork._sigmoid(X) * (1 - NeuralNetwork._sigmoid(X))

    @staticmethod
    def _to_class_id(Y_pred):
        """
        Create a m by 1 matrix with m the number of sample. Takes  as an input the Y_pred matrix createdFor each sample, the class id with most class id matrix for each sample the ass
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

    def _init_neural_network(self):
        """
        Initialiaze the network: allocate space memory for the matrices according to the network topology.
        Note that for convenience, self.z[0] and self.delta[0] are initialized although they are useless for the
        input layer. This allow a uniform definition of j for the layer number when calling the matrices.
        :return:
        """
        self.weight = []
        self.w_delta = []
        self.unit = []
        self.delta = []
        for l in range(0, len(self.topology)):
            self.unit.append(np.zeros(self.topology[l]))
            self.delta.append([np.zeros(self.topology[l])])
            if l < len(self.topology) - 1:
                self.weight.append(np.random.rand(self.topology[l + 1], self.topology[l] + 1))
                self.w_delta.append(np.zeros((self.topology[l + 1], self.topology[l] + 1)))

    def _compute_layer_val(self, j):
        """
        compute the neuron value from the previous layer
        a: vector of shape 1 + number of unit in layer j
        self.weight[j]: matrix of size (S(j+1), S(j) + 1) with S(j) number of neurons in layer jx
        :param j: layer number to be computed
        :return: z[j], unit[j]
        """
        a = np.insert(self.unit[j - 1], 0, 1)
        z = np.matmul(self.weight[j - 1], a)
        return NeuralNetwork._sigmoid(z)

    def _forward_propagation(self):
        for l in range(1, len(self.topology)):
            self.unit[l] = self._compute_layer_val(l)

    def _backpropagation(self, y):
        """

        :param y: expected value. Vector of size n with n the number of unit in the last layer.
        :return:
        """
        self.delta[-1] = self.unit[-1] - y
        for l in range(len(self.topology) - 2, 1, -1):
            self.delta[l] = np.matmul(self.delta[l + 1], self.weight[l])[1:] * (self.unit[l] * (1 - self.unit[l]))
            self.w_delta[l] += np.matmul(self.delta[l + 1].reshape(-1, 1), np.insert(self.unit[l], 0, 1).reshape(1, -1))

    def _update_weight(self, nb_of_sample):
        for l in range(len(self.topology) - 1):
            if self.regularization != 0:
                _lambda = np.diag(np.diag(np.ones((self.weight[l].shape[0], self.weight[l].shape[0]))) * self.regularization)
                _lambda[0, 0] = 0
            else:
                _lambda = np.zeros(self.w_delta.shape)
            w_grad = self.w_delta[l] / nb_of_sample + np.matmul(_lambda, self.weight[l])
            self.weight[l] -= self.learning_rate * w_grad

    def train(self, X, Y, verbose=1):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param Y: matrix of shape (n_samples, n_output)
        :param verbose:
        :return:
        """
        self._init_neural_network()
        for i in range(self.nb_iter):
            if i % 100 == 0:
                print("iteration: {}".format(i))
            for m in range(X.shape[0]):
                self.unit[0] = X[m]
                self._forward_propagation()
                self._backpropagation(Y[m])
            self._update_weight(X.shape[0])



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
