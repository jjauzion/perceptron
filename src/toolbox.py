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
