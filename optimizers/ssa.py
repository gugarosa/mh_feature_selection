import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SSA(Optimizer):
    """A SSA class, inherited from Optimizer.
    This is the designed class to define SSA-related
    variables and methods.
    References:
        S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
        Advances in Engineering Software (2017).
    """

    def __init__(self, algorithm='SSA', hyperparams=None):
        """Initialization method.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> SSA.')

        # Override its parent class with the receiving hyperparams
        super(SSA, self).__init__(algorithm)

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
        
    def _update(self, agents, best_agent, function, c1):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            c1 (float): Coefficient valor.
        """
        
        for i, agent in enumerate(agents):
            if i == 0:
                # Generating an uniform random number
                c2 = r.generate_uniform_random_number()
                
                # Generating an uniform random number
                c3 = r.generate_uniform_random_number()
                
                # Update the position of the leading salp by Eq. 3.1
                if c3 >= 0:
                    agents[i].position = best_agent.position + c1 * ((agents[i].ub - agents[i].lb) * c2 + agents[i].lb)
                else:
                    agents[i].position = best_agent.position - c1 * ((agents[i].ub - agents[i].lb) * c2 + agents[i].lb)
            else:
                # Update the position of the follower salp by Eq. 3.4
                agents[i].position = 1/2 * (agents[i].position + agents[i-1].position)
        
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
                
                c1 = 2 * np.exp(-4*t/space.n_iterations)**2

                # Updating agents
                self._update(space.agents, space.best_agent, function, c1)

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
