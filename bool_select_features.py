import argparse
import pickle

import numpy as np
import opytimizer.math.random as r

import models.classifiers as c
import models.heuristics as h
import models.transfers as f
import utils.loader as l
import utils.optimizer as o
import utils.targets as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Select features using a boolean meta-heuristic optimization approach.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['arcene', 'basehock', 'caltech101', 'coil20',
                                                                       'isolet', 'lung', 'madelon', 'mpeg7', 'mpeg7_BAS',
                                                                       'mpeg7_FOURIER', 'mushrooms', 'ntl-commercial',
                                                                       'ntl-industrial', 'orl', 'pcmac', 'phishing',
                                                                       'segment', 'semeion', 'sonar', 'spambase',
                                                                       'vehicle', 'wine'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['bmrfo', 'bpso'])

    parser.add_argument('clf', help='Classifier identifier', choices=['opf'])

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=3)

    parser.add_argument('-val_split', help='Percentage of the validation set after train/test split', type=float, default=0.25)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.2)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Defining the numpy seed
    np.random.seed(args.seed)

    # Loading the data
    X_train, X_val, _, Y_train, Y_val, _ = l.load_dataset(args.dataset, val_split=args.val_split,
                                                          test_split=args.test_split, seed=args.seed)

    # Gathering the classifier
    clf = c.get_clf(args.clf).obj

    # Initializes the optimization target
    opt_fn = t.bool_feature_selection(clf, X_train, Y_train, X_val, Y_val)

    # Gathering the optimizer (meta-heuristic) and its arguments
    mh = h.get_heuristic(args.mh).obj
    n_agents = args.n_agents
    n_iterations = args.n_iter
    n_variables = X_train.shape[1]
    hyperparams = h.get_heuristic(args.mh).hyperparams

    # Defining boolean-based hyperparameters
    hyperparams['c1'] = r.generate_binary_random_number(size=(n_variables, 1))
    hyperparams['c2'] = r.generate_binary_random_number(size=(n_variables, 1))
    hyperparams['S'] = r.generate_binary_random_number(size=(n_variables, 1))

    # Runs the optimization task
    history = o.bool_optimize(mh, opt_fn, n_agents, n_variables, n_iterations, hyperparams)

    # Dumps the object to file
    file_path = f'history/{args.dataset}_{args.val_split}_{args.test_split}_{args.mh}_{args.clf}_{n_agents}ag_{n_iterations}iter_{args.seed}.pkl'
    with open(file_path, 'wb') as output_file:
        # Dumps object to file
        pickle.dump(history, output_file)
