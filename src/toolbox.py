import numpy as np
import math


def count_vector(vector):
    count = 0
    for val in vector:
        try:
            if not np.isnan(val):
                count += 1
        except TypeError as err:
            print("val = ", val)
            print(err.message)
            exit(0)
    return count


def mean_vector(vector):
    _sum = 0
    nb_nan = 0
    for val in vector:
        if not np.isnan(val):
            _sum += val
        else:
            nb_nan += 1
    count = len(vector) - nb_nan
    return _sum / count if count > 0 else np.nan


def min_vector(vector):
    try:
        min_val = vector[0]
    except IndexError:
        return None
    for val in vector:
        if val < min_val:
            min_val = val
    return min_val


def max_vector(vector):
    try:
        max_val = vector[0]
    except IndexError:
        return None
    for val in vector:
        if val > max_val:
            max_val = val
    return max_val


def std_vector(vector):
    _mean = mean_vector(vector)
    _sum = 0
    nb_nan = 0
    for val in vector:
        if not np.isnan(val):
            _sum += (val - _mean) ** 2
        else:
            nb_nan += 1
    count = len(vector) - nb_nan
    return math.sqrt(_sum / (count - 1)) if count > 0 else np.nan


def percentile_vector(vector, centile):
    if 0 < centile > 100:
        raise ValueError("centile shall be between 0 and 100. Got '{}'".format(centile))
    if len(vector) <= 0:
        return None
    sorted_vect = np.sort(vector)
    sorted_vect = sorted_vect[~np.isnan(sorted_vect)]
    if len(sorted_vect) == 0:
        return np.nan
    index = (len(sorted_vect) - 1) * centile / 100
    decimal = index - math.floor(index)
    if decimal != 0:
        a = sorted_vect[math.floor(index)]
        b = sorted_vect[math.floor(index) + 1]
        return a + (b - a) * decimal
    else:
        return sorted_vect[int(index)]


def one_hot_encode(y):
    """
    Transform y vector with multi value to a Y matrix with 1 and 0 for each class:
    y = [1, 3, 0, 2]
    Y = [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 1, 0],
    :param y: vector of size m with m the nb of samples
    :return: Y matrix of size (m, nb_class)
    """
    return (np.arange(np.max(y) + 1) == y[:, np.newaxis]).astype(float)


def compute_accuracy(y, y_pred):
    nb_class = np.unique(y).shape[0]
    confusion_matrix = np.zeros((nb_class, nb_class), dtype=int)
    for i in range(y.shape[0]):
        confusion_matrix[int(y_pred[i]), int(y[i])] += 1
    total_predicted = np.sum(confusion_matrix, axis=1)
    total_true = np.sum(confusion_matrix, axis=0)
    true_positive = np.diagonal(confusion_matrix)
    precision = true_positive / total_true
    recall = np.zeros(true_positive.shape, dtype=float)
    np.divide(true_positive, total_predicted, out=recall, where=(total_predicted != 0))
    f1score = np.zeros(nb_class, dtype=float)
    tmp = precision + recall
    np.divide((2 * precision * recall), tmp, out=f1score, where=(tmp != 0))
    precision = np.insert(precision, 0, np.average(precision))
    recall = np.insert(recall, 0, np.average(recall))
    f1score = np.insert(f1score, 0, np.average(f1score))
    accuracy = np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]
    return confusion_matrix, precision, recall, f1score, accuracy


def print_accuracy(precision, recall, f1score, accuracy, nb_class, class_name=None, confusion_matrix=None):
    """
    Print model's performance scores (accuracy, recall, precision, F1score)
    :param class_name: list containing the name of each class in order
    """
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
        line[1] += "{}%".format(str(round(precision[i] * 100, 2))).ljust(col_padding[i + 1], " ")
        line[2] += "{}%".format(str(round(recall[i] * 100, 2))).ljust(col_padding[i + 1], " ")
        line[3] += "{}%".format(str(round(f1score[i] * 100, 2))).ljust(col_padding[i + 1], " ")
    print("\n".join(line))
    print("{title:<{width1}}{val:<{width2}}".format(
        title="Accuracy", width1=col_padding[0], val=str(round(accuracy * 100, 2)) + "%", width2=col_padding[1]))
    if confusion_matrix is not None:
        print("Confusion matrix:\n{}".format(confusion_matrix))

