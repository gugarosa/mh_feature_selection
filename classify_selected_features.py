import argparse

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

import models.classifiers as c
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Classify pre-selected features over the testing set.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['arcene', 'basehock', 'caltech101', 'coil20',
                                                                       'isolet', 'lung', 'madelon', 'mpeg7', 'mpeg7_BAS',
                                                                       'mpeg7_FOURIER', 'mushrooms', 'ntl-commercial',
                                                                       'ntl-industrial', 'orl', 'pcmac', 'phishing',
                                                                       'segment', 'semeion', 'sonar', 'spambase',
                                                                       'vehicle', 'wine'])

    parser.add_argument('clf', help='Classifier identifier', choices=['opf'])

    parser.add_argument('features', help='Selected features file', type=str)

    parser.add_argument('-val_split', help='Percentage of the validation set after train/test split', type=float, default=0.25)

    parser.add_argument('-test_split', help='Percentage of the testing set', type=float, default=0.2)

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
    X_train, _, X_test, Y_train, _, Y_test = l.load_dataset(args.dataset, val_split=args.val_split,
                                                            test_split=args.test_split, seed=args.seed)

    # Loading the selected features
    features = np.loadtxt(input_path + '_val.txt')[:-3].astype(bool)

    # Remaking training and testing sets with selected features
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]

    # Gathering the classifier and the transfer function
    clf = c.get_clf(args.clf).obj()

    # Fits training data into the classifier
    clf.fit(X_train_selected, Y_train)

    # Predicts new data
    preds = clf.predict(X_test_selected)

    # Calculating metrics
    acc = accuracy_score(Y_test, preds)
    f1 = f1_score(Y_test, preds, average='weighted')
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')

    # Saving final accuracy into an output file
    np.savetxt(input_path + '_test.txt', [acc, f1, precision, recall],
               header='Accuracy (1) | F1 (1) | Precision (1) | Recall (1)')
