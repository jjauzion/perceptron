import argparse
from pathlib import Path
import numpy as np

from src import check_arg
from src import wrapper_fct
from src import processing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-trf", "--train_file", type=str, default="data/data_train.csv", help="train dataset file, csv format")
    parser.add_argument("-tef", "--test_file", type=str, default="data/data_test.csv", help="test dataset file, csv format")
    parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
    parser.add_argument("-t", "--topology", type=check_arg.is_positive_int, nargs="+", default=[31, 31, 1], help="topology of the NN, ex: -t 4 4 1")
    parser.add_argument("-gc", "--grad_checking", action="store_true", help="Activate the gradient checking. Only for code debbuging.")
    parser.add_argument("-i", "--iteration", type=check_arg.is_positive_int, default=500, help="Number of iteration to run")
    parser.add_argument("-s", "--step", type=check_arg.is_positive_int, default=100, help="Step betweeen two print report")
    parser.add_argument("-m", "--model", type=str, help="load an existing model to pursue training")
    parser.add_argument("-a", "--act_fct", type=str, choices=["sigmoid", "softmax"], help="Activation function to be used in the output layer. Sigmoid by default")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=0, help="Level of verbosity")
    args = parser.parse_args()
    df_train = wrapper_fct.create_dataframe(args.train_file, not args.no_header, converts={1: ["B", "M"]})
    df_train.scale(exclude_col=1)
    df_train.save_scale_and_label(Path("model/data_train_scale.pkl"))
    df_train.describe()
    print("Nb de 1 : ", np.count_nonzero(df_train.data[:, 1] == 1))
    print("Nb de 0 : ", np.count_nonzero(df_train.data[:, 1] == 0))
    df_test = wrapper_fct.create_dataframe(args.test_file, not args.no_header, scale=Path("model/data_train_scale.pkl"))

    if args.model is None:
        model = processing.NeuralNetwork(topology=args.topology, regularization_rate=0,
                                         seed=4, model_name="m1", activation_fct=args.act_fct)
    else:
        model = processing.NeuralNetwork()
        model.load_model(args.model)
    nb_iter = args.iteration if args.iteration > 0 else "auto"
    if nb_iter == "auto":
        model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], gradient_checking=args.grad_checking,
                  verbose=args.verbose, nb_iteration=nb_iter)
    else:
        for i in range(args.step, nb_iter, args.step):
            model.fit(np.delete(df_train.data, 1, axis=1), df_train.data[:, 1], gradient_checking=args.grad_checking,
                      verbose=0, nb_iteration=args.step)
            test_f1score, test_loss = wrapper_fct.check_test(df=df_test, model=model, verbose=0)
            print("iteration:{} ; train_loss={:.3f} ; test_loss={:.3f} ; train_score={:.3f}% ; test_score={:.3f}%".format(
                model.nb_iteration_ran, model.cost_history[-1], test_loss, model.f1score[0] * 100, test_f1score * 100))
    if args.verbose > 1:
        model.plot_training()
    model.save_model(Path("model/m1.pkl"))
    wrapper_fct.check_test(df=df_test, model=model)
