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
        tmp = self.nb_class if self.nb_class > 1 else 2
        self.confusion_matrix = np.zeros((tmp, tmp), dtype=int)
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
        nb_class = self.nb_class if self.nb_class > 1 else 2
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

    def _to_class_id(self, Y_pred):
        """
        Create a m by 1 matrix with m the number of sample. Takes  as an input the Y_pred matrix createdFor each sample, the class id with most class id matrix for each sample the ass
        :param Y_pred: m by nb_class matrix, with m nb of sample
        :return: m by 1 matrix -> predicted class number
        """
        if self.nb_class > 1:
            return Y_pred.argmax(axis=1)
        else:
            return np.round(Y_pred).flatten()

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
        regul = self.regularization / (2 * X.shape[0]) * (np.sum(self.weight ** 2))
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
        # self.recall = true_positive / total_predicted
        # np.nan_to_num(self.recall, copy=False)
        self.recall = np.zeros(true_positive.shape, dtype=float)
        np.divide(true_positive, total_predicted, out=self.recall, where=(total_predicted != 0))
        self.recall = np.insert(self.recall, 0, np.average(self.recall))
        # self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)
        # np.nan_to_num(self.f1score, copy=False)
        self.f1score = np.zeros(self.precision.shape, dtype=float)
        tmp = self.precision + self.recall
        np.divide((2 * self.precision * self.recall), tmp, out=self.f1score, where=(tmp != 0))
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

    def __init__(self, nb_itertion=1000, learning_rate=0.1, regularization_rate=0, topology=None, model_name=None):
        """

        :param nb_itertion:
        :param learning_rate:
        :param regularization_rate:
        :param topology: list of size nb_of_layer (including input and output layer) with the nb of neuron for each
        layer : [nb_of_n_in_layer_0, nb_of_n_in_layer_1, ...,nb_of_n_in_layer_L]
        :param model_name:
        """
        self.topology = topology
        self.nb_iter = nb_itertion
        self.learning_rate = learning_rate
        self.regularization = 0 if regularization_rate is None else regularization_rate
        self.name = model_name
        self.confusion_matrix = np.zeros((self.topology[-1], self.topology[-1]), dtype=int)
        self.precision = [-1]
        self.recall = [-1]
        self.f1score = [-1]
        self.accuracy = -1
        self.weight = None
        self.unit = None
        self.delta = None
        self.w_delta = None
        self.cost_history = np.zeros((nb_itertion, self.topology[-1]))

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

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _derivative_sigmoid(X):
        return NeuralNetwork._sigmoid(X) * (1 - NeuralNetwork._sigmoid(X))

    @staticmethod
    def _to_class_id(Y_pred):
        """
        Create a m by 1 matrix with m the number of sample. Takes as an input the Y_pred matrix created.
        For each sample, the index of the class id with higher value (i.e. most probable class) is taken.
        Example:
        Y_pred = [[0.99, 0.1, 0], [0.01, 0.2, 0.8]] will return as [[0], [2]]
        :param Y_pred: m by nb_class matrix, with m nb of sample
        :return: m by 1 matrix -> predicted class number
        """
        return Y_pred.argmax(axis=1)

    def _compute_hypothesis_with_custom_weight(self, X, weight, layer):
        """
        Perform a forward propagation but using custom weight instead of weight stored in self.
        :param X:
        :param weight:
        :param layer:
        :return:
        """
        H = np.zeros((X.shape[0], self.topology[-1]))
        for m in range(X.shape[0]):
            h = np.copy(X[m])
            for l in range(1, len(self.topology)):
                a = np.insert(h, 0, 1)
                if l - 1 == layer:
                    z = np.matmul(weight, a)
                else:
                    z = np.matmul(self.weight[l - 1], a)
                h = NeuralNetwork._sigmoid(z)
            H[m] = h
        return H

    def _gradient_checking(self, X, Y, layer, w_grad):
        """
        Check the weight gradient (w_grad) computed from back propagation to a numerical gradient computation
        :param layer: int. Layer number to be checked
        :param w_grad: weight gradient for this layer compute from back propagation
        :return:
        """
        epsilon = 0.0001
        perturb = np.zeros(w_grad.shape)
        num_grad = np.zeros(w_grad.shape)
        it = np.nditer(w_grad, flags=['multi_index'])
        while not it.finished:
            perturb[it.multi_index[0], it.multi_index[1]] = epsilon
            H1 = self._compute_hypothesis_with_custom_weight(X, self.weight[layer] - perturb, layer)
            loss1 = self._compute_cost(Y, H1)
            H2 = self._compute_hypothesis_with_custom_weight(X, self.weight[layer] + perturb, layer)
            loss2 = self._compute_cost(Y, H2)
            num_grad[it.multi_index[0], it.multi_index[1]] = (loss2 - loss1) / (2 * epsilon)
            perturb[it.multi_index[0], it.multi_index[1]] = 0
            it.iternext()
        if not np.allclose(num_grad, w_grad):
            print("It seems there is an error in gradient computation...")
            print("w_grad =\n{}".format(w_grad))
            print("num_grad =\n{}".format(num_grad))
            print("diff =\n{}".format(abs(num_grad - w_grad)))
        else:
            print("All good !")
            print("w_grad =\n{}".format(w_grad))
            print("num_grad =\n{}".format(num_grad))
            print("diff =\n{}".format(abs(num_grad - w_grad)))

    def _compute_cost(self, Y, H):
        """
        Compute the cross entropy loss of prediction H versus Y true values

        cost formula:  -1/m * sum(sum(Y * log(H) + (1 - Y) * log(1 - H)))
        Note:          Z = np.sum(np.diag(X.dot(Y))) <=> Z = np.sum(X * Y.T)
        regul formula: lambda / (2m) * sum(sum(sum(Theta^2)))

        self.weight : n by nb_class matrix, with n the number of parameter
        :param Y: m by nb_class matrix, with m nb of sample
        :param H: m by nb_class matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return:
        """
        prod = np.row_stack(Y) * np.log(H)
        print("DEBUG\nY * log(H)\n{}".format(prod))
        prod = (1 - np.row_stack(Y)) * np.log(1 - H)
        print("DEBUG\n1-Y * log(1-H)\n{}".format(prod))
        cost = -1 / Y.shape[0] * (np.sum(np.row_stack(Y) * np.log(H) + (1 - np.row_stack(Y)) * np.log(1 - H)))
        regul = 0
        for l in range(len(self.topology) - 1):
            regul += np.sum(self.weight[l] ** 2)
        regul = regul * self.regularization / (2 * Y.shape[0])
        return cost + regul

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
        return self.unit[-1]

    def _backpropagation(self, y):
        """

        :param y: expected value. Vector of size n with n the number of unit in the last layer.
        :return:
        """
        self.delta[-1] = self.unit[-1] - y
        for l in range(len(self.topology) - 2, 0, -1):
            self.delta[l] = np.matmul(self.delta[l + 1], self.weight[l])[1:] * (self.unit[l] * (1 - self.unit[l]))
            self.w_delta[l] += np.matmul(self.delta[l + 1].reshape(-1, 1), np.insert(self.unit[l], 0, 1).reshape(1, -1))
        self.w_delta[0] += np.matmul(self.delta[1].reshape(-1, 1), np.insert(self.unit[0], 0, 1).reshape(1, -1))

    def _update_weight(self, nb_of_sample, gradient_checking=False, X=None, Y=None):
        """
        Compute the partial derivate of the cost with respect to each weight parameters (w_grad) and update the weight
        with these gradient as follow: weight = weight - learning_rate * w_grad
        :param nb_of_sample: number of sample in the training set. The weight delta has previously been computed over
        all of the training sample
        :param gradient_checking: If True, will compare the computed value of w_grad to a numerical approx of the gradient
        :param X: If gradient_checking is enabled, input X matrix shall be given (size: nb_of_sample by nb_of_input)
        :param Y: If gradient_checking is enabled, input Y matrix shall be given (size: nb_of_sample by nb_of_output)
        """
        for l in range(len(self.topology) - 1):
            regul = self.weight[l] * self.regularization
            regul[0] = 0
            w_grad = (self.w_delta[l] + regul) / nb_of_sample
            self.weight[l] -= self.learning_rate * w_grad
            if gradient_checking:
                if X is None or Y is None:
                    raise AttributeError("X and Y param are required for gradient checking")
                self._gradient_checking(X, Y, l, w_grad)

    def train(self, X, Y, verbose=1, gradient_checking=False):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param Y: matrix of shape (n_samples, n_output)
        :param gradient_checking: If True, enable the gradient checking at each iteration
        :param verbose:
        :return:
        """
        self._init_neural_network()
        self.cost_history = []
        y_pred = np.ones((X.shape[0], self.topology[-1])) * -1
        for i in range(self.nb_iter):
            if verbose >= 1 and i % 100 == 0:
                print("iteration: {}".format(i))
            for m in range(X.shape[0]):
                self.unit[0] = X[m]
                self._forward_propagation()
                y_pred[m] = self.unit[-1]
                self._backpropagation(Y[m])
            self._update_weight(X.shape[0], gradient_checking=gradient_checking, X=X, Y=Y)
            print("DEBUG\nY({}), y_pred({})".format(Y.shape, y_pred.shape))
            print("DEBUG\nY, y_pred\n{}".format(np.hstack((Y.reshape(X.shape[0], -1), y_pred.reshape(X.shape[0], -1)))))
            self.cost_history.append(self._compute_cost(Y=Y, H=y_pred))
        # Y_pred = self._compute_hypothesis(X)
        # y_pred = self._to_class_id(Y_pred)
        # self._compute_accuracy(y, y_pred)
        if verbose >= 1:
            print("Training completed!")
            # self.print_accuracy()
        if verbose >= 2:
            self.plot_training()

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
        if X.shape[1] + 1 != self.weight[0].shape[1]:
            raise ValueError("The input X matrix dimension ({}) (bias unit added) doesn't match with model weight shape ({})"
                             .format(X.shape + 1, self.weight[1].shape))
        y_pred = np.ones((X.shape[0], self.topology[-1])) * -1
        for m in range(X.shape[0]):
            self.unit[0] = X[m]
            y_pred[m] = self._forward_propagation()
        if verbose >= 1:
            print("Prediction completed!".format())
        if verbose >= 2:
            print(y_pred)
        return self._to_class_id(y_pred), y_pred

    def save_model(self, file):
        with Path(file).open(mode='wb') as fd:
            pickle.dump(self.__dict__, fd)
