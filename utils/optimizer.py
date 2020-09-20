from otimizador import Opytimizer
from otimizador.core.function import Function
from otimizador.core.optimizer import Optimizer
from otimizador.spaces.boolean import BooleanSpace
from otimizador.spaces.search import SearchSpace


def bool_optimize(opt, target, n_agents, n_variables, n_iterations, hyperparams):
    """Abstracts all boolean-based Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the BooleanSpace
    space = BooleanSpace(n_agents=n_agents, n_variables=n_variables, n_iterations=n_iterations)

    # Creating the Optimizer
    optimizer = opt(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializing task
    history = task.start(store_best_only=True)

    return history


def optimize(opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the SearchSpace
    space = SearchSpace(n_agents=n_agents, n_variables=n_variables, n_iterations=n_iterations,
                        lower_bound=lb, upper_bound=ub)

    # Creating the Optimizer
    optimizer = opt(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializing task
    history = task.start(store_best_only=True)

    return history
