import numpy as np


def process_history(history, transfer):
    """Process an optimization history object and saves its output as readable files.

    Args:
        history (opytimizer.utils.History): An optimization history.
        transfer (callable): A callable to the transfer function itself.

    """

    # Gathering the best agent's position as a numpy array
    features = np.asarray(history.best_agent[-1][0])

    # Passing it down the transfer function and transforms it into an integer
    selected_features = transfer(features).astype(int)

    # Gathering the number of selected features
    n_selected_features = np.count_nonzero(selected_features)

    # Gathering fitness
    fit = history.best_agent[-1][1]

    # Gathering optimization time
    time = history.time

    return selected_features, n_selected_features, fit, time
