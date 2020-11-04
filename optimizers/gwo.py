import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class GWO(Optimizer):
    """A GWO class, inherited from Optimizer.
    This is the designed class to define GWO-related
    variables and methods.
    References:
        S. Mirjalili and S. M. Mirjalili and A. Lewis.
        Grey Wolf Optimizer. Advances in Engineering Software (2014).
    """

    def __init__(self, algorithm='GWO', hyperparams=None):
        """Initialization method.
        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> GWO.')

        # Override its parent class with the receiving hyperparams
        super(GWO, self).__init__(algorithm)
        
        # Now, we need to build this class up
        self._build()

        logger.info('Class overrided.')
    
    def _build(self):
        """This method serves as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.
        """

        logger.debug('Running private method: build().')

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Built: %s.', self.algorithm, self.built)
    
    def _update(self, agents, function, alpha):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.
        """
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        best_1, best_2, best_3 = copy.deepcopy(agents[:3])
        
        # Iterates through all agents
        for agent in agents:
            
            A1 = alpha * (2 * r.generate_uniform_random_number() - 1)
            A2 = alpha * (2 * r.generate_uniform_random_number() - 1)
            A3 = alpha * (2 * r.generate_uniform_random_number() - 1)
            
            C1 = 2 * r.generate_uniform_random_number()
            C2 = 2 * r.generate_uniform_random_number()
            C3 = 2 * r.generate_uniform_random_number()
            
            X1 = best_1.position - A1 * abs(C1 * best_1.position - agent.position)
            X2 = best_2.position - A2 * abs(C2 * best_2.position - agent.position)
            X3 = best_3.position - A3 * abs(C3 * best_3.position - agent.position)
        
            temp = (X1 + X2 + X3) / 3.0
            fit  = function(temp)

            if fit < agent.fit:
                # Updates the corresponding agent's position
                agent.position = copy.deepcopy(temp)
                        
                # And its fitness as well
                agent.fit = copy.deepcopy(fit)

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
                
                alpha = 2 - 2 * t / (space.n_iterations - 1)

                # Updating agents
                self._update(space.agents, function, alpha)

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
