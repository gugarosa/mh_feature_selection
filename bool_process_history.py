import argparse

import numpy as np

import utils.outputter as o
from opytimizer.utils.history import History


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Process a boolean-based optimization history into readable outputs.')

    parser.add_argument('input_file', help='Optimization history file', type=str)

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

    # Process the optimization history
    features, n_features, fit, time = o.process_history(history, None)

    # Opening the output .txt file
    with open(input_path + '_val.txt', 'w') as output_file:
        # Saving selected features
        np.savetxt(output_file, features,
                   header=f'Selected_Features ({features.shape[0]}) | N_Selected_Features (1) | Fitness (1) | Time (1)')

        # Saving the number of selected features
        np.savetxt(output_file, [n_features])

        # Saving fitness
        np.savetxt(output_file, [fit])

        # Saving optimization time
        np.savetxt(output_file, [time])
