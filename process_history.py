import argparse

import numpy as np
from opytimizer.utils.history import History

import models.transfers as f
import utils.outputter as o


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Process an optimization history into readable outputs.')

    parser.add_argument('input_file', help='Optimization history file', type=str)

    parser.add_argument('transfer', help='Transfer function identifier', choices=['t1'])

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input parsed arguments
    args = get_arguments()

    # Defining the input file path
    input_path = f'history/{args.input_file}'

    # Creating an empty History object
    history = History()

    # Loading history from input file
    history.load(input_path + '.pkl')

    # Gathering the used transfer function
    transfer = f.get_transfer(args.transfer).obj

    # Process the optimization history
    features, n_features, fit, time = o.process_history(history, transfer)

    # Opening the output .txt file
    with open(input_path + '_val.txt', 'w') as output_file:
        # Saving selected features
        np.savetxt(output_file, features)

        # Saving the number of selected features
        np.savetxt(output_file, [n_features])

        # Saving fitness
        np.savetxt(output_file, [fit])

        # Saving optimization time
        np.savetxt(output_file, [time])
