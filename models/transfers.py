import math
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


def S1(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = 1.0 / (1.0 + np.exp(-1 * x))

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def S2(x):
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


def S3(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = 1.0 / (1.0 + np.exp(-1 * x / 2))

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def S4(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = 1.0 / (1.0 + np.exp(-1 * x / 3))

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def V1(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = [np.fabs(math.erf(np.sqrt(np.pi) / 2 * (-1 * x[i]))) for i in range(len(x))]

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def V2(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = y = [np.fabs(math.tanh(-1 * x[i])) for i in range(len(x))]

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def V3(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = np.abs(-1 * x / np.sqrt(1 + (-1 * x*x)))

    # Rounding the transfer function and converting it to boolean
    y = np.round(y).astype(bool)

    return y


def V4(x):
    """Performs a transform that maps continuous to binary data.

    Args:
        x (np.array): An array of continuous values.

    Returns:
        An array of binary values.

    """

    # Passes down the transfer function
    y = [np.abs(2 / np.pi * math.atan(np.pi / 2 * (-1 * x[i]))) for i in range(len(x))]

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
    t1=Transfer(T1),
    s1=Transfer(S1),
    s2=Transfer(S2),
    s3=Transfer(S3),
    s4=Transfer(S4),
    v1=Transfer(V1),
    v2=Transfer(V2),
    v3=Transfer(V3),
    v4=Transfer(V4),
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
