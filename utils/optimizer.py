from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import BooleanSpace, SearchSpace, TreeSpace


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

    # Creates space, optimizer and function
    space = BooleanSpace(n_agents, n_variables)
    optimizer = opt(hyperparams)
    function = Function(target)

    # Creates the optimization task
    task = Opytimizer(space, optimizer, function)

    # Initializes the task
    task.start(n_iterations)

    return task.history


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

    # Creates space, optimizer and function
    space = SearchSpace(n_agents, n_variables, lb, ub)
    optimizer = opt(hyperparams)
    function = Function(target)

    # Creates the optimization task
    task = Opytimizer(space, optimizer, function)

    # Initializes the task
    task.start(n_iterations)

    return task.history


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

    # Creates space, optimizer and function
    space = TreeSpace(n_agents, n_variables, lb, ub, 5, 2, 5, ['SUM', 'SUB', 'MUL', 'DIV'])
    optimizer = opt(hyperparams)
    function = Function(target)

    # Creates the optimization task
    task = Opytimizer(space, optimizer, function)

    # Initializes the task
    task.start(n_iterations)

    return task.history
