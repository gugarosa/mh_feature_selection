from opytimizer.optimizers.swarm import pso


class Heuristic:
    """An Heuristic class helps users in selecting distinct optimization heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines an heuristic dictionary constant with the possible values
HEURISTIC = dict(
    pso=Heuristic(pso.PSO, dict(w=0.7, c1=1.7, c2=1.7))
)


def get_heuristic(name):
    """Gets an heuristic by its identifier.

    Args:
        name (str): Heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return HEURISTIC[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Heuristic {name} has not been specified yet.')
