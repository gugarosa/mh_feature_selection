import numpy as np
import opytimizer.math.random as r

def T1(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    #
    r1 = r.generate_uniform_random_number(size=x.shape[0])

    #
    y = 1.0 / (1.0 + np.exp(-2 * x))

    #
    binary_y = np.where(r1 < y, 1, 0).astype(bool)

    return binary_y
