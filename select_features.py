import argparse

import numpy as np

import models.classifiers as c
import models.heuristics as h
import opt.target as t
import opt.wrapper as w
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Select features using a meta-heuristic optimization approach.')

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['pso'])

    parser.add_argument('clf', help='Classifier identifier', choices=['lr'])

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Defining the numpy seed
    np.random.seed(args.seed)

    # Loading the data
    X_train, X_val, Y_train, Y_val = l.load_dataset()

    # Gathering the classifier
    clf = c.get_clf(args.clf).obj

    # Initializes the optimization target
    opt_fn = t.feature_selection(clf, X_train, Y_train, X_val, Y_val)

    # Gathering the optimizer (meta-heuristic) and its arguments
    mh = h.get_heuristic(args.mh).obj
    n_agents = args.n_agents
    n_iterations = args.n_iter
    n_variables = X_train.shape[1]
    lb = [-1] * n_variables
    ub = [1] * n_variables
    hyperparams = h.get_heuristic(args.mh).hyperparams

    # Runs the optimization task
    history = w.optimize(mh, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)
