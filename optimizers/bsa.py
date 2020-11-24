"""Backtracking Search Optimization Algorithm.
"""

import copy
import math

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer
from opytimizer.spaces.search import SearchSpace

logger = l.get_logger(__name__)


class BSA(Optimizer):
    """A BSA class, inherited from Optimizer.

    This is the designed class to define BSA-related
    variables and methods.

    References:
        P. Civicioglu, “Backtracking search optimization algorithm for numerical
        optimization problems,” Applied Mathematics and Computation, vol.
        219, no. 15, pp. 8121–8144, 2013.

    """

    def __init__(self, algorithm='BSA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(BSA, self).__init__(algorithm)

        # Controls the number of elements of individuals that will mutate in a trial
        self.mix_rate = 1.0

        # Controls the amplitude of the search-direction matrix
        self.F = 3.0
        
        # Historical population
        self.historicalPopularion = None

        # Now, we need to build this class up
        self._build(hyperparams)       

        logger.info('Class overrided.')
        
    @property
    def historicalPopularion(self):
        """list: list of past agents.

        """

        return self._historicalPopularion

    @historicalPopularion.setter
    def historicalPopularion(self, historicalPopularion):
        self._historicalPopularion = historicalPopularion

    @property
    def mix_rate(self):
        """float: Mix Rate.

        """

        return self._mix_rate

    @mix_rate.setter
    def mix_rate(self, mix_rate):
        if not isinstance(mix_rate, (float, int)):
            raise e.TypeError('`mix_rate` should be a float or integer')
        self._mix_rate = mix_rate

    @property
    def F(self):
        """float: Controls the amplitude of the search-direction matrix.

        """

        return self._F

    @F.setter
    def F(self, F):
        if not isinstance(F, (float, int)):
            raise e.TypeError('`F` should be a float or integer')

        self._F = F

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
            if 'mix_rate' in hyperparams:
                self.mix_rate = hyperparams['mix_rate']
            if 'F' in hyperparams:
                self.F = hyperparams['F']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Hyperparameters: mix_rate = %s, F = %s| Built: %s.',
                     self.algorithm, self.mix_rate, self.F, self.built)
        
    def _boundaryControlMechanism(self, agents, crossOver):
        """Mutates a new agent based on pre-picked distinct agents.

        Args:
            agents (List): Set of Agents.
            crossOver (Searchspace): Searchspace comprising crossovered agents.

        """
        for i, agent in enumerate(agents):
            for j in range(agent.n_variables):
                for k in range(agent.n_dimensions):
                    if (crossOver.agents[i].position[j,k] < agent.lb[j] or (crossOver.agents[i].position[j,k] > agent.ub[j])):
                        crossOver.agents[i].position[j,k] = ( (agent.ub[j] - agent.lb[j]) * r.generate_uniform_random_number()) + agent.lb[j]    
	    
    def _permutation(self):
        """Performs the permutation of the historical population's agents.

        """
        for i in range(len(self.historicalPopularion)):
            #generate a random position
            temp_i = r.generate_integer_random_number(low=0, high=len(self.historicalPopularion))
            if (temp_i!=i):
                tmp = copy.deepcopy(self.historicalPopularion[i].position)
                self.historicalPopularion[i].position = copy.deepcopy(self.historicalPopularion[temp_i].position)
                self.historicalPopularion[temp_i].position= copy.deepcopy(tmp)
                
    def _initializeMutation(self, space):
        """Initializes a set of mutant agents.

        Args:
            space (Searchspace): Current search space.

        Returns:
            A set of mutated agent.

        """
        mutation = np.zeros((space.n_agents,space.n_variables, space.agents[0].n_dimensions))
        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                for k in range(agent.n_dimensions):
                    mutation[i,j,k] =  agent.position[j,k] + (self.F*r.generate_uniform_random_number()*(self.historicalPopularion[i].position[j,k] -  agent.position[j,k]));
        return mutation	    
	    
    def _generateMap(self, space):
        """Generates a map to agents to be changed.

        Args:
            space (Searchspace): Current search space.

        Returns:
            A map to agents to be changed.

        """
        Map = np.zeros((space.n_agents,space.n_variables))
        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                Map[i,j] = 1;
                
        # number of permutations 
        if r.generate_uniform_random_number() <r.generate_uniform_random_number():
            MaxU = math.ceil(self._mix_rate * space.n_variables * r.generate_uniform_random_number());
            for i, agent in enumerate(space.agents):	
                for j in range(1,MaxU):
                    u = r.generate_integer_random_number(low=0, high=space.n_variables)  
                    Map[i][u] = 0;

        else:
            for i, agent in enumerate(space.agents):	
                randi = r.generate_integer_random_number(low=0, high=space.n_variables)  
                Map[i][randi] = 0;  
                
        return Map

	    
    def _crossOver(self, space, mutation):
        """Performs the crossover operation.

        Args:
            space (Searchspace): Current search space.
            mutation (List): List of mutated agents.

        Returns:
            A trial population

        """
    
        Map = self._generateMap(space);        

        trialPopulation = SearchSpace(n_agents=space.n_agents, n_variables=space.n_variables, n_iterations=space.n_iterations,
                        lower_bound=space.lb, upper_bound=space.ub)  

        # Generation of a trial population 
        for i, agent in enumerate(space.agents):	
            for j in range(space.n_variables):
                trialPopulation.agents[i].position[j] = copy.deepcopy( mutation[i,j])   
                
        for i, agent in enumerate(space.agents):
            for j in range(space.n_variables):
                if Map[i,j]:
                    trialPopulation.agents[i].position[j] = copy.deepcopy(agent.position[j])

        return trialPopulation

	    
    def _update(self, space, function):
        """Method that wraps selection and mutation updates over all
        agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.

        """
        agents = space.agents

        # SELECTION - I
        if r.generate_uniform_random_number() <r.generate_uniform_random_number():
            for i, agent in enumerate(agents):
                self.historicalPopularion[i].position = copy.deepcopy(agent.position)
        self._permutation()  

        # Mutation     
        mutation = self._initializeMutation(space)


        # Cross Over - generates Trial-population		
        trialPopulation = self._crossOver(space, mutation)

        # Check agent limits
        self._boundaryControlMechanism(agents, trialPopulation)



        # SELECTION - II 
        # Calculates the fitness for trial population
        self._evaluate(trialPopulation, function)   

        # Iterate through all agents
        for i, agent in enumerate(agents):
                
            if trialPopulation.agents[i].fit <agent.fit:
                agent.position = copy.deepcopy(trialPopulation.agents[i].position)    
                agent.fit = copy.deepcopy(trialPopulation.agents[i].fit)


            if agent.fit < space.best_agent.fit:
                # It updates the global best value and position 
                space.best_agent.position = copy.deepcopy(agent.position)  
                space.best_agent.fit = copy.deepcopy(agent.fit)              

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
        
        # initializes a historical population
        self.historicalPopularion = SearchSpace(n_agents=space.n_agents, n_variables=space.n_variables, n_iterations=space.n_iterations,
                        lower_bound=space.lb, upper_bound=space.ub).agents   
                        
        

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
                self._update(space, function)

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
