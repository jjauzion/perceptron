from pathlib import Path
import numpy as np
import multiprocessing
import os

from src import check_test
from src import processing
from src import dataframe


def train_model(id_nb):
    topology = str(nb_input) + "," + "{},".format(topology_range["max_unit"]) * l + str(u) + "," + str(nb_output)
    model_name = "model_{}_r{}".format(topology, r)
    ppid = os.getppid()
    log = "{}|----------------------\n{}|{}\n".format(ppid, ppid, model_name)
    model = processing.NeuralNetwork(topology=[int(e) for e in topology.split(",")], regularization_rate=r, model_name=model_name, seed=4)
    delta_cost = 100
    total_iter = 0
    test_score = []
    train_score = []
    while (delta_cost > 0.01 or total_iter < 3000) and total_iter < 10000:
        model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], verbose=0, nb_iteration=100)
        total_iter += step
        delta_cost = (model.cost_history[-2] - model.cost_history[-1]) * 100 / model.cost_history[-1]
        test_score.append(check_test.check_test(df_tool, df_file=test_file, model=model, verbose=0))
        train_score.append(model.f1score[0])
        log += "{}|i={} ; f1score train={} test={} ; delta_cost={}\n".format(ppid, total_iter, train_score[-1], test_score[-1], delta_cost)
    model.save_model(Path("model/{}.pkl".format(model_name)))
    convergence = model.cost_history[::step]
    model_stat = np.array([test_score, train_score, convergence]).T
    np.savetxt("model/stat_id{}_{}.csv".format(id_nb, model_name), model_stat, delimiter=",", header="test_score,train_score,cost")
    print(log)
    return [id_nb, test_score[-1], train_score[-1], np.prod([int(e) for e in topology.split(",")]), r, model.nb_iteration_ran]


if __name__ == "__main__":
    train_file = "data/data_train.csv"
    test_file = "data/data_test.csv"
    nb_input = 31
    nb_output = 1
    topology_range = {
        "max_hidden_layer": 2,
        "max_unit": 31,
        "min_unit": 27,
        "unit_step": 4
    }
    # regul_range = [0, 0.01, 0.1, 1, 10]
    regul_range = [0]
    df_tool = "model/data_train_scale.pkl"

    df_train = dataframe.DataFrame()
    try:
        df_train.read_from_csv(train_file, header=False, converts={1: ["B", "M"]})
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
        print("Could not read file '{}' because : {}".format(Path(train_file), err))
        exit(0)
    df_train.scale(exclude_col=1)
    df_train.save_scale_and_label(Path(df_tool))
    l_unit = list(range(topology_range["min_unit"], topology_range["max_unit"] + 1, topology_range["unit_step"]))
    stat = np.zeros((topology_range["max_hidden_layer"] * len(l_unit) * len(regul_range), 6))
    step = 100
    id_nb = -1
    proc = {}
    for l in range(topology_range["max_hidden_layer"]):
        for u in l_unit:
            for r in regul_range:
                id_nb += 1
                proc["parent_connec"], proc["child_connec"] = multiprocessing.Pipe()

    header = "id,test_score,train_score,polynomial_degree,regul_rate,nb_iterations"
    np.savetxt("model/stat.csv", stat, delimiter=",", header=header)

