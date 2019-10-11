from pathlib import Path
import numpy as np

from . import check_test
from . import processing
from . import dataframe

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
df_tool = "model/data_train_scale.pkl"

df_train = dataframe.DataFrame()
try:
    df_train.read_from_csv(train_file, header=False, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(train_file), err))
    exit(0)
df_train.scale(exclude_col=1)
df_train.save_scale_and_label(Path(df_tool))
for l in range(topology_range["max_hidden_layer"]):
    for u in range(topology_range["min_unit"], topology_range["max_unit_in_hidden"] + 1, topology_range["unit_step"]):
        for r in regul_range:
            topology = str(nb_input) + str(topology_range["max_unit"]) * l + str(u) + str(nb_output)
            model_name = "model_{}_r{}".format(topology, r)
            model = processing.NeuralNetwork(topology=topology.split(","), regularization_rate=r, model_name=model_name, seed=4)
            delta_cost = 100
            total_iter = 0
            test_score = []
            train_score = []
            while delta_cost > 0.01 and total_iter < 10000:
                model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], verbose=0, nb_iteration=100)
                total_iter += 100
                delta_cost = (model.cost_history[-2] - model.cost_history[-1]) * 100 / model.cost_history[-1]
                test_score.append(check_test.check_test(df_tool, df_file=test_file, model=model))
                train_score.append(model.f1score[0])
            model.save_model(Path("model/m1.pkl"))
            model_list = {
                model_name: {
                    "test_score": test_score,
                    "train_score": train_score,
                    "polynomial_degree": np.prod(topology.split(",")),
                    "regul_rate": r
                }
            }

