from pathlib import Path
import numpy as np
import multiprocessing as mp
import os

from src import wrapper_fct
from src import processing


def train_model(task_input):
    model_id = task_input[0]
    model_topology = task_input[1]
    model_regul_rate = task_input[2]
    df_train = task_input[3]
    df_test = task_input[4]
    model_name = "model_{}_r{}".format(model_topology, model_regul_rate)
    pid = os.getpid()
    log = "{}|----------------------\n{}|model:'{}' ; id:'{}'\n".format(pid, pid, model_name, model_id)
    model = processing.NeuralNetwork(topology=[int(e) for e in model_topology.split(",")], regularization_rate=model_regul_rate, model_name=model_name, seed=4)
    delta_cost = 100
    total_iter = 0
    test_score = []
    train_score = []
    step = 100
    # while (delta_cost > 0.01 or total_iter < 3000) and total_iter < 10000:
    while (delta_cost > 1 or total_iter < 300) and total_iter < 10000:
        model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], verbose=0, nb_iteration=step)
        total_iter += step
        delta_cost = (model.cost_history[-2] - model.cost_history[-1]) * 100 / model.cost_history[-1]
        test_score.append(wrapper_fct.check_test(df=df_test, model=model, verbose=0)[0])
        train_score.append(model.f1score[0])
        log += "{}|i={} ; f1score train={} test={} ; delta_cost={}\n".format(pid, total_iter, train_score[-1], test_score[-1], delta_cost)
        print(log)
    model.save_model(Path("model/{}.pkl".format(model_name)))
    convergence = model.cost_history[::step]
    model_stat = np.array([test_score, train_score, convergence]).T
    print(model_stat)
    np.savetxt("model/stat_id{}_{}.csv".format(model_id, model_name), model_stat, delimiter=",", header="test_score,train_score,cost")
    print(log)
    return [model_id, model_topology, test_score[-1], train_score[-1], np.prod([int(e) for e in model_topology.split(",")]), model_regul_rate, model.nb_iteration_ran]


if __name__ == "__main__":
    train_file = "data/data_train.csv"
    test_file = "data/data_test.csv"
    nb_input = 31
    nb_output = 1
    topology_range = {
        "max_hidden_layer": 1,
        "max_unit": 31,
        "min_unit": 27,
        "unit_step": 4
    }
    # regul_range = [0, 0.01, 0.1, 1, 10]
    regul_range = [0]
    df_tool = "model/data_train_scale.pkl"

    df_train = wrapper_fct.create_dataframe(train_file, header=False, converts={1: ["B", "M"]})
    df_train.scale(exclude_col=1)
    df_train.save_scale_and_label(Path(df_tool))
    df_test = wrapper_fct.create_dataframe(test_file, header=False, scale=Path(df_tool))
    df_test.scale(exclude_col=1)
    l_unit = list(range(topology_range["min_unit"], topology_range["max_unit"] + 1, topology_range["unit_step"]))
    stat = np.zeros((topology_range["max_hidden_layer"] * len(l_unit) * len(regul_range), 6))
    id_nb = -1
    task = []
    for l in range(topology_range["max_hidden_layer"]):
        for u in l_unit:
            for r in regul_range:
                id_nb += 1
                topology = str(nb_input) + "," + "{},".format(topology_range["max_unit"]) * l + str(u) + "," + str(nb_output)
                task.append((id_nb, topology, r, df_train, df_test))
    print("nb of task = ", len(task))
    pool = mp.Pool(2)
    stat = np.array(pool.map(train_model, task))
    header = "id,topology,test_score,train_score,polynomial_degree,regul_rate,nb_iterations"
    np.savetxt("model/stat.csv", X=stat, delimiter=",", header=header, fmt="%s")

