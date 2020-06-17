import numpy as np


def T1(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = 1.0 / (1.0 + np.exp(-2 * x))

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


class Transfer:
    """A Transfer class helps users in selecting distinct transfer functions from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (callable): A callable instance.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a transfer function dictionary constant with the possible values
TRANSFER = dict(
    t1=Transfer(T1)
)


def get_transfer(name):
    """Gets a transfer function by its identifier.

    Args:
        name (str): Transfer function's identifier.

    Returns:
        A callable that refers to the transfer function itself.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return TRANSFER[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Transfer function {name} has not been specified yet.')
