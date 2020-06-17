import argparse

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import models.classifiers as c
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Classify baseline features over the testing set.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['wine'])

    parser.add_argument('clf', help='Classifier identifier', choices=['lr'])

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
    X_train, _, X_test, Y_train, _, Y_test = l.load_dataset(args.dataset, val_split=args.val_split,
                                                            test_split=args.test_split, seed=args.seed)

    # Gathering the classifier and the transfer function
    clf = c.get_clf(args.clf).obj()

    # Fits training data into the classifier
    clf.fit(X_train, Y_train)

    # Predicts new data
    preds = clf.predict(X_test)

    # Calculating metrics
    acc = accuracy_score(Y_test, preds)
    f1 = f1_score(Y_test, preds, average='weighted')
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')

    # Saving final accuracy into an output file
    np.savetxt(f'history/{args.dataset}_{args.val_split}_{args.test_split}_{args.seed}_test.txt', [acc, f1, precision, recall],
               header='Accuracy (1) | F1 (1) | Precision (1) | Recall (1)')
