import numpy as np


def process_history(history, transfer):
    """Process an optimization history object and saves its output as readable files.

    Args:
        history (opytimizer.utils.History): An optimization history.
        transfer (callable): A callable to the transfer function itself.

    """

    # Gathering the best agent's position as a numpy array
    best = np.asarray(history.best_agent[-1][0])

    # Gathering fitness as well
    fit = history.best_agent[-1][1]

    # Passing it down the transfer function and transforms it into an integer
    transferred_best = transfer(best).astype(int)

    return transferred_best, fit
