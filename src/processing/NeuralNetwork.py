import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from . import Model
from .. import toolbox


class NeuralNetwork(Model.Classification):
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

    def __init__(self, topology=None, nb_iteration=1000, learning_rate=0.1, regularization_rate=0, model_name=None, seed=None):
        """

        :param nb_iteration:
        :param learning_rate:
        :param regularization_rate:
        :param topology: list of size nb_of_layer (including input and output layer) with the nb of neuron for each
        layer : [nb_of_n_in_layer_0, nb_of_n_in_layer_1, ...,nb_of_n_in_layer_L]
        :param model_name:
        :param seed: seed to be used for the random initialisation of the weights.
        """
        self.topology = topology if topology is not None else [2, 1]
        self.weight = None
        self.w_delta = None
        self.unit = None
        self.delta = None
        Model.Classification.__init__(self, nb_iteration, learning_rate, regularization_rate, self.topology[-1],
                                      model_name)
        self.init_neural_network(seed)

    def init_neural_network(self, seed=None):
        """
        Initialiaze the network: allocate space memory for the matrices according to the network topology.
        Note that for convenience, self.z[0] and self.delta[0] are initialized although they are useless for the
        input layer. This allow a uniform definition of j for the layer number when calling the matrices.
        :param seed: seed value to be used for the random initialisation of the weights
        """
        self.weight = []
        self.w_delta = []
        self.unit = []
        self.delta = []
        for l in range(0, len(self.topology)):
            self.unit.append(np.zeros(self.topology[l]))
            self.delta.append([np.zeros(self.topology[l])])
            if l < len(self.topology) - 1:
                if seed is not None:
                    np.random.seed(seed + l)
                self.weight.append(np.random.rand(self.topology[l + 1], self.topology[l] + 1))
                self.w_delta.append(np.zeros((self.topology[l + 1], self.topology[l] + 1)))

    def describe(self):
        """Print model characterisic"""
        print("Model: {}".format(self.name))
        print("Topology: {}".format(self.topology))
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

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

    def _gradient_checking(self, X, Y, layer, w_grad, rtol=0.1, atol=0.005):
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
        if not np.allclose(num_grad, w_grad, rtol=rtol, atol=atol):
            print("WARNING: It seems there is an error in gradient computation...")
            print("w_grad =\n{}".format(w_grad))
            print("num_grad =\n{}".format(num_grad))
            print("diff =\n{}".format(abs(num_grad - w_grad) - atol + rtol * abs(w_grad)))

    def _compute_cost(self, Y, H):
        """
        Compute the cross entropy loss of prediction H versus Y true values

        cost formula:  -1/m * sum(sum(Y * log(H) + (1 - Y) * log(1 - H)))
        Note:          Z = np.sum(np.diag(X.dot(Y))) <=> Z = np.sum(X * Y.T)
        regul formula: lambda / (2m) * sum(sum(sum(Theta^2)))

        self.weight : n by nb_output_unit matrix, with n the number of parameter
        :param Y: m by nb_output_unit matrix, with m nb of sample
        :param H: m by nb_output_unit matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :return:
        """
        cost = -1 / Y.shape[0] * (np.sum(np.row_stack(Y) * np.log(H) + (1 - np.row_stack(Y)) * np.log(1 - H)))
        regul = 0
        for l in range(len(self.topology) - 1):
            regul += np.sum(self.weight[l] ** 2)
        regul = regul * self.regularization / (2 * Y.shape[0])
        return cost + regul

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

    def fit(self, X, y, nb_iteration=None, verbose=1, gradient_checking=False):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param y: vector of size n_samples
        :param nb_iteration: number of iteration to run. If None, will take number stored in self.nb_iteration
        :param gradient_checking: If True, enable the gradient checking at each iteration
        :param verbose:
        :return:
        """
        if nb_iteration is not None:
            self.nb_iter = nb_iteration
        self.cost_history = []
        Y = toolbox.one_hot_encode(y, self.nb_output) if self.nb_output > 1 else y.reshape(-1, 1)
        Y_pred = np.ones((X.shape[0], self.topology[-1])) * -1
        for i in range(self.nb_iter):
            if verbose >= 1 and i % 100 == 0:
                print("iteration: {}".format(i))
            for l in range(len(self.topology) - 1):
                self.w_delta[l][:] = 0
            for m in range(X.shape[0]):
                self.unit[0] = X[m]
                self._forward_propagation()
                Y_pred[m] = self.unit[-1]
                self._backpropagation(Y[m])
            self._update_weight(X.shape[0], gradient_checking=gradient_checking, X=X, Y=Y)
            self.cost_history.append(self._compute_cost(Y=Y, H=Y_pred))
        y_pred = self._to_class_id(Y_pred)
        self.compute_accuracy(y, y_pred)
        if verbose >= 1:
            print("Training completed!")
            self.print_accuracy()
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
