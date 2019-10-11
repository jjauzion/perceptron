from pathlib import Path
import numpy as np

from src import check_test
from src import processing
from src import dataframe

train_file = "data/data_train.csv"
test_file = "data/data_test.csv"
nb_input = 32
nb_output = 1
topology_range = {
    "max_hidden_layer": 2,
    "max_unit": 32,
    "min_unit": 4,
    "unit_step": 4
}
regul_range = [0, 0.01, 0.1, 1, 10]

df_train = dataframe.DataFrame()
df_test = dataframe.DataFrame()
try:
    df_train.read_from_csv(train_file, header=False, converts={1: ["B", "M"]})
    df_test.read_from_csv(test_file, header=False, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(train_file), err))
    exit(0)
df_train.scale(exclude_col=1)
df_train.save_scale_and_label(Path("model/data_train_scale.pkl"))
df_test.scale(exclude_col=1)
for l in range(topology_range["max_hidden_layer"]):
    for u in range(topology_range["min_unit"], topology_range["max_unit_in_hidden"] + 1, topology_range["unit_step"]):
        for r in regul_range:
            topology = str(nb_input) + str(topology_range["max_unit"]) * l + str(u) + str(nb_output)
            model_name = "model_{}_r{}".format(topology, r)
            model = processing.NeuralNetwork(topology=topology.split(","), regularization_rate=r, model_name=model_name, seed=4)
            model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], verbose=0, nb_iteration="auto")
            model.save_model(Path("model/m1.pkl"))
            model_list = {model_name: model}
check_test.check_test("model/data_train_scale.pkl", df_file="data/data_test.csv", model=model)

