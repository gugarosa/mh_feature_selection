import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class CSA(Optimizer):
    """A CSA class, inherited from Optimizer.
    This is the designed class to define CSA-related
    variables and methods.
    References:
        A. Askarzadeh. A novel metaheuristic method for
        solving constrained engineering optimization problems: Crow search algorithm.
        Computers & Structures (2016).
    """

    def __init__(self, algorithm='CSA', hyperparams=None):
        """Initialization method.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> CSA.')

        # Override its parent class with the receiving hyperparams
        super(CSA, self).__init__(algorithm)
        
        # Flight length
        self.fl = 2
        
        # Awareness probability
        self.AP = 0.1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def fl(self):
        """float: flight length.
        """

        return self._fl

    @fl.setter
    def fl(self, fl):
        if not isinstance(fl, (float, int)):
            raise e.TypeError('`fl` should be a float or integer')

        self._fl = fl
        
    @property
    def AP(self):
        """float: awareness probability.
        """

        return self._AP

    @AP.setter
    def AP(self, AP):
        if not isinstance(AP, (float, int)):
            raise e.TypeError('`AP` should be a float or integer')
        if AP < 0 or AP > 1:
            raise e.ValueError('`AP` should be between 0 and 1')

        self._AP = AP
        
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
            if 'fl' in hyperparams:
                self.fl = hyperparams['fl']
            if 'AP' in hyperparams:
                self.AP = hyperparams['AP']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s| Hyperparameters: fl = %s, AP = %s | '
                     'Built: %s.',
                     self.algorithm, self.fl, self.AP, self.built)
    
    @d.pre_evaluation
    def _evaluate(self, space, function, memory):
        """Evaluates the search space according to the objective function.
        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            memory (np.array): Array of memories.
        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the memory to current's agent position
                memory[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position to the best agent
                space.best_agent.position = copy.deepcopy(memory[i])

                # Makes a deep copy of current agent fitness to the best agent
                space.best_agent.fit = copy.deepcopy(agent.fit)
    
    def _update(self, agents, best_agent, function, memory):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            memory (np.array): Array of memories.
        """
        
        for agent in agents:
            
            # Generating an integer random number
            j = r.generate_integer_random_number(0, len(agents))
            
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()
            
            # Generating an uniform random number
            r2 = r.generate_uniform_random_number()
            
            if r1 > self.AP:
                agent.position = agent.position + r2 * self.fl * (memory[j] - agent.position)
            else:
                # Generate a new agent's position
                for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                    # For each decision variable, we generate uniform random numbers
                    agent.position[j] = r.generate_uniform_random_number(lb, ub, size=agent.n_dimensions)
            
        
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
        
        # Instanciating array of memories
        memory = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, memory, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')
                
                # Updating agents
                self._update(space.agents, space.best_agent, function, memory)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, memory, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history
