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


class SOS(Optimizer):
    """An SOS class, inherited from Optimizer.
    This is the designed class to define SOS-related
    variables and methods.
    References:
        M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
        Computers & Structures (2014).
    """

    def __init__(self, algorithm='SOS', hyperparams=None):
        """Initialization method.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> SOS.')

        # Override its parent class with the receiving hyperparams
        super(SOS, self).__init__(algorithm)

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
    
    def _mutualism(self, agent_i, agent_j, best_agent, function):
        
        # Creates a temporary agent_i and agent_j
        a = copy.deepcopy(agent_i)
        b = copy.deepcopy(agent_j)
        
        mutual_vector = (agent_i.position + agent_j.position) / 2
        
        BF_1, BF_2 = np.random.choice([1, 2], 2, replace=False)
        
        # Generating an uniform random number
        r1 = r.generate_uniform_random_number()
        
        a.position = agent_i.position + r1 * (best_agent.position - mutual_vector * BF_1)
        
        b.position = agent_j.position + r1 * (best_agent.position - mutual_vector * BF_2)
        
        # Checking agents limits
        a.clip_limits()
        b.clip_limits()
        
        # Evaluates the agent
        a.fit = function(a.position)
        b.fit = function(b.position)
        
        # If the new potision is better than the current agent's position
        if a.fit < agent_i.fit:
            # Replace the current agent's position
            agent_i.position = copy.deepcopy(a.position)

            # Also replace its fitness
            agent_i.fit = copy.deepcopy(a.fit)
            
        # If the new potision is better than the current agent's position
        if b.fit < agent_j.fit:
            # Replace the current agent's position
            agent_j.position = copy.deepcopy(b.position)

            # Also replace its fitness
            agent_j.fit = copy.deepcopy(b.fit)
    
    def _commensalism(self, agent_i, agent_j, best_agent, function):
        
        # Creates a temporary agent_i and agent_j
        a = copy.deepcopy(agent_i)
        
        # Generating an uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)
        
        a.position = agent_i.position + r1 * (best_agent.position - agent_j.position)
        
        # Checking agents limits
        a.clip_limits()
        
        # Evaluates the agent
        a.fit = function(a.position)
        
        # If the new potision is better than the current agent's position
        if a.fit < agent_i.fit:
            # Replace the current agent's position
            agent_i.position = copy.deepcopy(a.position)

            # Also replace its fitness
            agent_i.fit = copy.deepcopy(a.fit)
    
    def _parasitism(self, agent_i, agent_j, best_agent, function):
        
        # Creates a temporary agent_i and agent_j
        parasite = copy.deepcopy(agent_i)
        
        # Generating an uniform random number
        r1 = r.generate_integer_random_number(0, agent_i.n_variables)
        
        parasite.position[r1] = r.generate_uniform_random_number(agent_i.lb, agent_i.ub)
        
        # Checking agents limits
        parasite.clip_limits()
        
        # Evaluates the agent
        parasite.fit = function(parasite.position)
        
        # If the new potision is better than the current agent's position
        if parasite.fit < agent_j.fit:
            # Replace the current agent's position
            agent_j.position = copy.deepcopy(parasite.position)

            # Also replace its fitness
            agent_j.fit = copy.deepcopy(parasite.fit)
    
    def _update(self, agents, best_agent, function):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
        """
        
        for agent in agents:
            # Generating an integer random number
            j = r.generate_integer_random_number(0, len(agents))
            
            self._mutualism(agent, agents[j], best_agent, function)
            
            # Generating an integer random number
            j = r.generate_integer_random_number(0, len(agents))
        
            self._commensalism(agent, agents[j], best_agent, function)
            
            # Generating an integer random number
            j = r.generate_integer_random_number(0, len(agents))
        
            self._parasitism(agent, agents[j], best_agent, function)
        
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
                
                # Updating agents
                self._update(space.agents, space.best_agent, function)

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
