import numpy as np
from pathlib import Path
import pickle


class IdentityScale:
    """
    No scale, return data as is.
    """
    name = "IdentityScale"

    def __repr__(self):
        return "{}".format(self.name)

    def fit(self, data):
        return True

    def transform(self, data):
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class MeanNormScaler:

    name = "MeanNormScale"

    def __init__(self):
        self.mean = None
        self.scale = None

    def __repr__(self):
        return "{} ; mean = {} ; std = {}".format(self.name, self.mean, self.scale)

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.mean = np.average(data, axis=0)
        self.scale = np.std(data, axis=0)

    def transform(self, data, inplace=False):
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.mean[i]) / self.scale[i]
        if inplace:
            data[:] = normalized_data[:]
        return normalized_data

    def fit_transform(self, data, inplace=False):
        self.fit(data)
        return self.transform(data, inplace=inplace)

    def load(self, file):
        with Path(file).open(mode='rb') as fp:
            scale = pickle.load(fp)
        self.mean = scale["mean"]
        self.scale = scale["scale"]

    def save(self, file):
        scale = {
            "mean": self.mean,
            "scale": self.scale
        }
        with Path(file).open(mode='wb') as fp:
            pickle.dump(scale, fp)


class MinMaxScaler:

    name = "MeanNormScale"

    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def __repr__(self):
        return "{} ; min = {} ; max = {}".format(self.name, self.min, self.max)

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.min = np.nanmin(data, axis=0)
        self.max = np.nanmax(data, axis=0)
        self.range = self.max - self.min

    def transform(self, data, inplace=False):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.min[i]) / self.range[i]
        if inplace:
            data[:] = normalized_data[:]
        return normalized_data

    def fit_transform(self, data, inplace=False):
        self.fit(data)
        return self.transform(data,inplace=inplace)

    def load(self, file):
        with Path(file).open(mode='rb') as fp:
            scale = pickle.load(fp)
        self.min = scale["min"]
        self.max = scale["max"]
        self.range = scale["range"]

    def save(self, file):
        scale = {
            "min": self.min,
            "max": self.max,
            "range": self.range
        }
        with Path(file).open(mode='wb') as fp:
            pickle.dump(scale, fp)


class LabelEncoder:

    def __init__(self, classes=None):
        self._class = []
        if classes is not None:
            self.fit(classes)

    def fit(self, classes):
        self._class = list(set(classes))

    def transform(self, classes, ignore_unknown_class=True):
        try:
            if isinstance(classes, list):
                label = [float(self._class.index(elm)) for elm in classes]
            else:
                if isinstance(classes, bytes):
                    classes = classes.decode('utf-8')
                label = float(self._class.index(classes))
        except ValueError:
            if ignore_unknown_class:
                return np.nan
            else:
                raise ValueError("One of the given name is not a valid class name.\nGot '{}' ; Valid class name : '{}'"
                                 .format(classes, self._class))
        except TypeError:
            raise TypeError("Input given is not an iterable.\nGot'{}'".format(type(classes)))
        return label

    def fit_transform(self, classes):
        self.fit(classes)
        return self.transform(classes)

    def inverse_transform(self, label):
        try:
            classes = [self._class[int(index)] for index in label]
        except IndexError:
            raise IndexError("One of the label is out of range. Got '{}'".format(label))
        return classes
