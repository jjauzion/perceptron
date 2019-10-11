from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from . import toolbox as tb
from . import preprocessing
from . import plotlib


class DataFrame:

    def __init__(self, import_scale_and_label=None, data=None):
        """
        Create a dataframe object.
        :param import_scale_and_label:
        :param data: If data is provided, the dataframe is initiazed with data. Data shall be a matrix of shape m by n
        where m is the number of sample and n the number of features.
        """
        if data is not None:
            self.data = data.astype(float)
            self.header = np.arange(data.shape[1])
        else:
            self.data = None
            self.header = []
        self.original_data = None
        self.scaler = None
        self.labelizer = None
        if import_scale_and_label is not None:
            self.load_scale_and_label(import_scale_and_label)

    # MAP & REDUCE FUNCTION

    def count(self, axis=0):
        return np.apply_along_axis(tb.count_vector, axis=axis, arr=self.data)

    def mean(self, axis=0):
        return np.apply_along_axis(tb.mean_vector, axis=axis, arr=self.data)

    def min(self, axis=0):
        return np.apply_along_axis(tb.min_vector, axis=axis, arr=self.data)

    def max(self, axis=0):
        return np.apply_along_axis(tb.max_vector, axis=axis, arr=self.data)

    def std(self, axis=0):
        return np.apply_along_axis(tb.std_vector, axis=axis, arr=self.data)

    def percentile(self, centile, axis=0):
        return np.apply_along_axis(tb.percentile_vector, axis, self.data, centile)

    def count_nan(self, axis=0):
        return np.sum(np.isnan(self.data), axis=axis)

    # DATA ANALYSIS FUNCTION

    def describe(self, floating_point=2):
        self.count_nan()
        stats = np.array([
            self.count(), self.mean(), self.std(), self.min(),
            self.percentile(25), self.percentile(50), self.percentile(75), self.max(), self.count_nan()
        ])
        stats_df = pd.DataFrame(stats, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'NaN'],
                                columns=self.header)
        stats_df.dropna(axis='columns', inplace=True)
        pd.options.display.width = 0
        print(stats_df.round(floating_point))

    def scatter(self, ycol, xcol=None, color_col=None):
        plotlib.scatter(self.data, ycol, x_column=xcol, y_title=self.header[ycol],
                        x_title=self.header[xcol] if xcol is not None else "x", color_col=color_col)

    def pair_plot(self, hue_col=None):
        plotlib.pair_plot(self.data, self.header, hue_col)

    # DATAFRAME MODIFICATION FUNCTION

    def scale(self, scale_type="minmax", first_col=0, exclude_col=None):
        """

        :param scale_type: minmax (default) or meannorm
        :param first_col: nb of column at the beginning of the df_train that shall not be scaled
        :param exclude_col: int -> index of one column to exclude of the scaling
        :return:
        """
        if self.scaler is None:
            if scale_type == "minmax":
                self.scaler = preprocessing.MinMaxScaler()
            elif scale_type == "meannorm":
                self.scaler = preprocessing.MeanNormScaler()
            else:
                raise ValueError("scale type unknown. Got '{}'".format(scale_type))
        if exclude_col is None:
            self.scaler.fit_transform(self.data[:, first_col:], inplace=True)
        else:
            self.data = np.insert(
                self.scaler.fit_transform(np.delete(self.data[:, first_col:], exclude_col, axis=1)),
                exclude_col,
                self.data[:, exclude_col],
                axis=1
            )

    def drop_column(self, col_index):
        self.data = np.delete(self.data, col_index, axis=1)
        self.header = np.delete(self.header, col_index)

    def drop_nan_column(self):
        mask = ~np.all(np.isnan(self.data), axis=0)
        self.data = self.data[:, mask]
        self.header = self.header[mask]

    def drop_nan_row(self):
        self.data = self.data[~np.any(np.isnan(self.data), axis=1)]

    # SAVE & LOAD

    def read_from_csv(self, file, header=True, converts=None):
        """
        create a dataframe from a csv file
        :param file: csv file path to open
        :param header: if True (default), the first line is read as a header to get column names
        :param converts: {column: [classes]} -> will convert the value in column: each value of 'classes' will be a numeric value
        """
        with Path(file).open(mode='r', encoding='utf-8') as fp:
            if header:
                line = fp.readline()
                if len(line) == 0:
                    raise ValueError("File '{}' seems to be empty...".format(file))
                self.header = np.array(line[:-1].split(",") if line[-1] == "\n" else line.split(","))
            if converts is not None:
                if self.labelizer is not None:
                    print("Warning: previous labelizer has been overwritten with the given converts argument")
                converters = {}
                self.labelizer = {}
                for column in converts:
                    self.labelizer[column] = preprocessing.LabelEncoder(converts[column])
                    converters[column] = self.labelizer[column].transform
            elif self.labelizer is not None:
                converters = {}
                for column in self.labelizer.keys():
                    converters[column] = self.labelizer[column].transform
            else:
                converters = None
            self.data = np.genfromtxt(fp, delimiter=',', dtype="float64", converters=converters, missing_values="?")
            if len(self.header) == 0:
                self.header = np.arange(self.data.shape[1])
        self.original_data = np.copy(self.data)

    def save_scale_and_label(self, file):
        df_tool = {
            "scaler": self.scaler,
            "labelizer": self.labelizer
        }
        with Path(file).open(mode='wb') as fp:
            pickle.dump(df_tool, fp)

    def load_scale_and_label(self, file):
        with Path(file).open(mode='rb') as fp:
            try:
                df_tool = pickle.load(fp)
                self.labelizer = df_tool["labelizer"]
                self.scaler = df_tool["scaler"]
            except (pickle.UnpicklingError, EOFError, TypeError, IndexError, KeyError, ValueError) as err:
                raise ValueError("Can't load scale and label from '{}' because : {}".format(file, err))
        if not isinstance(self.labelizer, dict):
            raise ValueError("Given file '{}' is not well formatted to load scale and label".format(file))
        if not (isinstance(self.scaler, preprocessing.MeanNormScaler) or isinstance(self.scaler, preprocessing.MinMaxScaler)):
            raise ValueError("Given file '{}' is not well formatted to load scale and label".format(file))

    def save(self, file):
        dataframe = {
            "data": self.data,
            "original_data": self.original_data,
            "header": self.header,
            "scaler": self.scaler,
            "labelizer": self.labelizer
        }
        with Path(file).open(mode='wb') as fp:
            pickle.dump(dataframe, fp)

    def load(self, file):
        with Path(file).open(mode='rb') as fp:
            try:
                dataframe = pickle.load(fp)
                self.labelizer = dataframe["labelizer"]
                self.scaler = dataframe["scaler"]
                self.header = dataframe["header"]
                self.original_data = dataframe["original_data"]
                self.data = dataframe["data"]
            except (pickle.UnpicklingError, EOFError, TypeError, IndexError, KeyError, ValueError) as err:
                raise ValueError("Can't load dataframe from '{}' because : {}".format(file, err))
        if not isinstance(self.labelizer, dict):
            raise ValueError("Given file '{}' is not well formatted: labelizer type '{}' is not a valid type".format(file, type(self.labelizer)))
        if not (isinstance(self.scaler, preprocessing.MeanNormScaler) or isinstance(self.scaler, preprocessing.MinMaxScaler)):
            raise ValueError("Given file '{}' is not well formatted: scaler type '{}' is not a valid type".format(file, type(self.scaler)))
