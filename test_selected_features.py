import argparse

import numpy as np
from sklearn.metrics import accuracy_score

import models.classifiers as c
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Test pre-selected features over the testing set.')

    parser.add_argument('clf', help='Classifier identifier', choices=['lr'])

    parser.add_argument('features', help='Selected features file', type=str)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Defining the input file path
    input_path = f'history/{args.features}'

    # Defining the numpy seed
    np.random.seed(args.seed)

    # Loading the data
    X_train, X_val, Y_train, Y_val = l.load_dataset()

    # Loading the selected features
    features = np.loadtxt(input_path + '_val.txt')[:-1].astype(bool)

    # Remaking training and testing sets with selected features
    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    # Gathering the classifier and the transfer function
    clf = c.get_clf(args.clf).obj()

    # Fits training data into the classifier
    clf.fit(X_train_selected, Y_train)

    # Predicts new data
    preds = clf.predict(X_val_selected)

    # Calculating accuracy
    acc = accuracy_score(Y_val, preds)

    # Saving final accuracy into an output file
    np.savetxt(input_path + '_test.txt', [acc])
