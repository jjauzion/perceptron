import argparse
from pathlib import Path
import numpy as np

from src import check_arg
from src import check_test
from src import processing
from src import dataframe

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="file to describe, shall be csv format")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
parser.add_argument("-t", "--topology", type=check_arg.is_positive_int, nargs="+", default=[31, 31, 1], help="topology of the NN, ex: -t 4 4 1")
parser.add_argument("-gc", "--grad_checking", action="store_true", help="Activate the gradient checking. Only for code debbuging.")
parser.add_argument("-i", "--iteration", type=check_arg.is_positive_int, default=500, help="Number of iteration to run")
parser.add_argument("-m", "--model", type=str, help="load an existing model to pursue training")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=not args.no_header, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.scale(exclude_col=1)
df.describe()
print("Nb de 1 : ", np.count_nonzero(df.data[:, 1] == 1))
print("Nb de 0 : ", np.count_nonzero(df.data[:, 1] == 0))

# args.model = "model/m1.pkl"
# args.topology = [31, 31, 1]
# args.iteration = 200

if args.model is None:
    model = processing.NeuralNetwork(topology=args.topology, regularization_rate=0,
                                     seed=4, model_name="m1")
else:
    model = processing.NeuralNetwork()
    model.load_model(args.model)
model.fit(np.delete(df.data, 1, axis=1), df.data[:, 1], gradient_checking=args.grad_checking, verbose=2, nb_iteration=args.iteration)
model.save_model(Path("model/m1.pkl"))
df.save_scale_and_label(Path("model/data_train_scale.pkl"))
check_test.check_test("model/data_train_scale.pkl", df_file="data/data_test.csv", model=model)

