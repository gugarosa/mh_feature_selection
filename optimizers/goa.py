import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.general as g
import opytimizer.utils.constants as c
import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class GOA(Optimizer):
    """A GOA class, inherited from Optimizer.
    This is the designed class to define GOA-related
    variables and methods.
    References:
        S. Saremi and S. Mirjalili and and A. Lewis.
        Grasshopper Optimisation Algorithm: Theory and application. Advances in Engineering Software (2017).
    """

    def __init__(self, algorithm='GOA', hyperparams=None):
        """Initialization method.
        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> GOA.')

        # Override its parent class with the receiving hyperparams
        super(GOA, self).__init__(algorithm)
        
        # Minimum comfort zone
        self.c_min = 0.0004

        # Maximum comfort zone
        self.c_max = 1
        
        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def c_min(self):
        """float: Minimum comfort zone range.
        """

        return self._c_min

    @c_min.setter
    def c_min(self, c_min):
        if not isinstance(c_min, (float, int)):
            raise e.TypeError('`c_min` should be a float or integer')
        if c_min < 0:
            raise e.ValueError('`c_min` should be >= 0')

        self._c_min = c_min

    @property
    def c_max(self):
        """float: Maximum comfort zone range.
        """

        return self._c_max

    @c_max.setter
    def c_max(self, c_max):
        if not isinstance(c_max, (float, int)):
            raise e.TypeError('`c_max` should be a float or integer')
        if c_max < 0:
            raise e.ValueError('`c_max` should be >= 0')
        if c_max < self.c_min:
            raise e.ValueError('`c_max` should be >= `c_min`')

        self._c_max = c_max
        
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
            if 'c_min' in hyperparams:
                self.c_min = hyperparams['c_min']
            if 'c_max' in hyperparams:
                self.c_max = hyperparams['c_max']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s| Hyperparameters: c_min = %s, c_max = %s | '
                     'Built: %s.',
                     self.algorithm, self.c_min, self.c_max, self.built)
    
    def _s(self, f, l, r):
        """Method that defines the strength of social forces.
        Args:
            f (float): Intensity of attraction.
            l (float): Attractive length scale.
            r (float): distance. 
        """
        
        return f * np.exp(-r / l) - np.exp(-r)
    
    def _update(self, agents, best_agent, function, comfort):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            comfort (float): A comfort zone parameter.
        """
        
        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(agents)
        
        # Iterating through 'i' agents
        for agent in agents:
            
            sum = np.zeros((agent.n_variables, agent.n_dimensions))
            
            # Iterating through 'j' agents
            for temp in temp_agents:
                
                # Distance is calculated by an euclidean distance between 'i' and 'j'
                distance = g.euclidean_distance(agent.position, temp.position)
                
                sum += (comfort / 2) * (np.expand_dims(agent.ub, -1) - np.expand_dims(agent.lb, -1)) * \
                        self._s(1.5, 0.5, np.linalg.norm(temp.position - agent.position, ord=1)) * \
                        (temp.position - agent.position) / (distance + c.EPSILON)
            
            # Updating position according to Equation 2.7
            agent.position = comfort * sum + best_agent.position
            
            # Checks agent limits
            agent.clip_limits()

            # Evaluates agent
            agent.fit = function(agent.position)    

    
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
                
                # Updating c according to Equation 2.8
                comfort = self.c_max - t * ((self.c_max - self.c_min) / space.n_iterations)

                # Updating agents
                self._update(space.agents, space.best_agent, function, comfort)

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
