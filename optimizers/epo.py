import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class EPO(Optimizer):
    """An EPO class, inherited from Optimizer.
    This is the designed class to define EPO-related
    variables and methods.
    References:
        G. Dhiman and V. Kumar. Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
        Knowledge-Based Systems (2018).
    """

    def __init__(self, algorithm='EPO', hyperparams=None):
        """Initialization method.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> EPO.')

        # Override its parent class with the receiving hyperparams
        super(EPO, self).__init__(algorithm)
        
        # Control parameter [2, 3]
        self.f = 2
        
        # Control parameter [1.5, 2]
        self.l = 1.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
    
    @property
    def f(self):
        """float: control parameter.
        """

        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, (float, int)):
            raise e.TypeError('`f` should be a float or integer')

        self._f = f
    
    @property
    def l(self):
        """float: control parameter.
        """

        return self._l

    @l.setter
    def l(self, l):
        if not isinstance(l, (float, int)):
            raise e.TypeError('`l` should be a float or integer')

        self._l = l
        
    def _build(self, hyperparams):
        """This method serves as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'f' in hyperparams:
                self.f = hyperparams['f']
            if 'l' in hyperparams:
                self.l = hyperparams['l']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s| Hyperparameters: f = %s, l = %s | '
                     'Built: %s.',
                     self.algorithm, self.f, self.l, self.built)
        
    def _update(self, agents, best_agent, function, T_s, t):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            T_s (float): Temperature profile.
            t (int): current iteration.
        """
        
        for agent in agents:
            
            # Eq. 10
            P_grid = np.abs(best_agent.position - agent.position)
            
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()
            
            # Eq. 9
            A = 2 * (T_s + P_grid) * r1 - T_s
            
            # Generating an uniform random number
            C = r.generate_uniform_random_number(0, 1, size = agent.n_variables)

            # Eq. 12
            S_A = np.abs(self.f * np.exp(-t / self.l) - np.exp(-t))
            
            # Eq. 8
            D_ep = np.abs(S_A * best_agent.position - C * agent.position)
            
            # Update the agent position based on Eq. 13
            agent.position = best_agent.position - A * D_ep
        
        
    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.
        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.
        Returns:
            A History object holding all agents' positions and fitness achieved during the task.
        """

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')
                
                # T = 0 if R > 1 else 1
                T_s = 1 - space.n_iterations / (t - space.n_iterations)
                
                # Updating agents
                self._update(space.agents, space.best_agent, function, T_s, t)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history
