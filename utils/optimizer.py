from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.spaces.boolean import BooleanSpace
from opytimizer.spaces.search import SearchSpace
from opytimizer.spaces.tree import TreeSpace


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


def optimize_with_gp(opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts Genetic Programming Opytimizer's mechanisms into a single method.

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

    # Creating the TreeSpace
    space = TreeSpace(n_trees=n_agents, n_terminals=5, n_variables=n_variables,
                      n_iterations=n_iterations, min_depth=2, max_depth=5,
                      functions=['SUM', 'SUB', 'MUL', 'DIV'], lower_bound=lb, upper_bound=ub)

    # Creating the Optimizer
    optimizer = opt(hyperparams=hyperparams)

    # Creating the Function
    function = Function(pointer=target)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializing task
    history = task.start(store_best_only=True)

    return history
