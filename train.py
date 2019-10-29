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
    parser.add_argument("-t", "--topology", type=check_arg.is_positive_int, nargs="+", default=[31, 16, 2], help="topology of the NN, ex: -t 4 4 1")
    parser.add_argument("-gc", "--grad_checking", action="store_true", help="Activate the gradient checking. Only for code debbuging.")
    parser.add_argument("-i", "--iteration", type=check_arg.is_positive_int, default=500, help="Number of iteration to run")
    parser.add_argument("-s", "--step", type=check_arg.is_positive_int, default=100, help="Step betweeen two print report")
    parser.add_argument("-lr", "--learning_rate", type=check_arg.is_positive, default=0.1, help="learning rate")
    parser.add_argument("-m", "--model", type=str, help="load an existing model to pursue training")
    parser.add_argument("-n", "--name", type=str, default="model", help="name of the model. Will be used to save the model to a pkl file")
    parser.add_argument("-a", "--act_fct", type=str, choices=["sigmoid", "softmax"], default="softmax", help="Activation function to be used in the output layer. Sigmoid by default")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=1, help="Level of verbosity")
    parser.add_argument("-tcol", "--target_col", type=check_arg.is_positive_int, default=1, help="Index of the column containing the target for prediction")
    parser.add_argument("-b", "--batch_size", type=check_arg.is_positive_int, default=0, help="Size of the mini batch. Set to 1 for Stochastic Gradient.")
    args = parser.parse_args()
    df_train = wrapper_fct.create_dataframe(args.train_file, not args.no_header, converts={args.target_col: ["B", "M"]})
    try:
        df_train.scale(exclude_col=args.target_col)
        df_train.save_scale_and_label(Path("model/data_train_scale.pkl"))
    except IndexError as err:
        print("Can't scale '{}' data because : {}".format(args.train_file, err))
        exit(0)
    df_test = wrapper_fct.create_dataframe(args.test_file, not args.no_header, scale=Path("model/data_train_scale.pkl"))
    try:
        df_test.scale(exclude_col=args.target_col)
    except IndexError as err:
        print("Can't scale '{}' data because : {}".format(args.train_file, err))
        exit(0)
    if df_train.data.shape[1] != df_test.data.shape[1]:
        print("Error: Number of column differs between test dataset ({}) and train dataset ({})".format(
            args.test_file, args.train_file))
        exit(0)
    if args.topology[0] <= 0:
        args.topology[0] = df_train.data.shape[1] - 1
    if df_train.data.shape[1] - 1 != args.topology[0]:
        print("Error: Topology first layer input number (={}) doesn't match the number of features in the dataset (={})".
              format(args.topology[0], df_train.data.shape[1] - 1))
        exit(0)
    if args.act_fct == "softmax" and args.topology[-1] < 2:
        print("Error: softmax activation function requires at least two units in the output layer.")
        exit(0)
    if args.topology[-1] > 2:
        print("Error: last layer cannot have more unit than number of classes")
        exit(0)

    if args.verbose > 0:
        print("Train dataset synthesis:")
        df_train.describe()
        print("Prediction column stat: {} samples are class '1' and {} samples are class '0'\n".format(
            np.count_nonzero(df_train.data[:, args.target_col] == 1),
            np.count_nonzero(df_train.data[:, args.target_col] == 0)))
    if args.model is None:
        model = processing.NeuralNetwork(topology=args.topology, regularization_rate=0,
                                         seed=4, model_name=args.name, activation_fct=args.act_fct,
                                         learning_rate=args.learning_rate)
    else:
        model = wrapper_fct.load_model(args.model)
        args.name = model.name
    if args.verbose > 0:
        print("Start training...")
    if args.iteration == 0:
        model.fit(np.delete(df_train.data, args.target_col, axis=1),
                  df_train.data[:, args.target_col],
                  gradient_checking=args.grad_checking, verbose=args.verbose, nb_iteration="auto",
                  batch_size=args.batch_size)
    else:
        for i in range(args.step, args.iteration + args.step, args.step):
            model.fit(np.delete(df_train.data, args.target_col, axis=1),
                      df_train.data[:, args.target_col],
                      gradient_checking=args.grad_checking, verbose=0, nb_iteration=args.step,
                      batch_size=args.batch_size)
            if args.verbose > 0:
                test_f1score, test_loss = wrapper_fct.check_test(target_col=args.target_col, df=df_test, model=model, verbose=0)
                delta_cost = (model.cost_history[-2] - model.cost_history[-1]) * 100 / model.cost_history[-1]
                print("iteration:{} ; train_loss={:.6f} ; test_loss={:.3f} ; train_score={:.3f}% ; test_score={:.3f}% ;"
                      " delta_loss = {:.6f} ".format(model.nb_iteration_ran, model.cost_history[-1], test_loss,
                                                     model.f1score[0] * 100, test_f1score * 100, delta_cost))
    model.save_model(Path("model/{}.pkl".format(args.name)))
    if args.verbose > 0:
        print("Training completed.\nModel saved to '{}'\n".format(Path("model/{}.pkl".format(args.name))))
    if args.verbose > 0:
        print("Score on test set after training:")
        wrapper_fct.check_test(target_col=args.target_col, df=df_test, model=model)
    if args.verbose > 1:
        model.plot_training()
