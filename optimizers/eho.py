import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class EHO(Optimizer):
    """A EHO class, inherited from Optimizer.
    This is the designed class to define EHO-related
    variables and methods.
    References:
        Gai-Ge Wang and S. Deb and and L. S. Coelho.
        Elephant Herding Optimization. 3rd International Symposium on Computational and Business Intelligence (2015).
    """

    def __init__(self, algorithm='EHO', hyperparams=None):
        """Initialization method.
        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> EHO.')

        # Override its parent class with the receiving hyperparams
        super(EHO, self).__init__(algorithm)
        
        # Matriarch influence scale factor
        self.alpha = 0.5

        # X_center influence scale factor
        self.beta = 0.1

        # Maximum number of clans
        self.n_clans = 5
        
        # Number of elephants in each clan
        self.nci = 10
        
        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def alpha(self):
        """float: Matriarch influence scale factor.
        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0 or alpha > 1:
            raise e.ValueError('`alpha` should be between 0 and 1')

        self._alpha = alpha

    @property
    def beta(self):
        """float: X_center influence scale factor.
        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0 or beta > 1:
            raise e.ValueError('`beta` should be between 0 and 1')

        self._beta = beta
        
    @property
    def n_clans(self):
        """int: Maximum number of clans.
        """

        return self._n_clans

    @n_clans.setter
    def n_clans(self, n_clans):
        if not isinstance(n_clans, int):
            raise e.TypeError('`n_clans` should be integer')
        if n_clans < 0:
            raise e.ValueError('`n_clans` should be >= 0')

        self._n_clans = n_clans
        
    @property
    def nci(self):
        """int: Number of elephants in each clan.
        """

        return self._nci

    @nci.setter
    def nci(self, nci):
        if not isinstance(nci, int):
            raise e.TypeError('`nci` should be integer')
        if nci < 0:
            raise e.ValueError('`nci` should be >= 0')

        self._nci = nci
        
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
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'n_clans'in hyperparams:
                self.n_clans = hyperparams['n_clans']
            if 'nci'in hyperparams:
                self.nci = hyperparams['nci']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s| Hyperparameters: alpha = %s, beta = %s, n_clans = %s, nci = %s | '
                     'Built: %s.',
                     self.algorithm, self.alpha, self.beta, self.n_clans, self.nci, self.built)
    
    def _create_clans(self, agents):
        
        clans = []
        
        for i in range(0, self.n_clans):
            ci = [copy.deepcopy(agents[j]) for j in range(0, self.nci)]
            clans.append(ci)
        
        return clans

    @d.pre_evaluation
    def _evaluate(self, space, clans, function):
        
        # Iterates through all clans
        for ci in clans:
            # Iterates through all agents
            for agent in ci:
                
                # Calculates the fitness value of current agent
                fit = function(agent.position)
                
                # If fitness is better than agent's best fit
                if fit < agent.fit:
                    # Updates its current fitness to the newer one
                    agent.fit = fit
                
                # If agent's fitness is better than global fitness
                if agent.fit < space.best_agent.fit:
                    # Makes a deep copy of agent's local best position to the best agent
                    space.best_agent.position = copy.deepcopy(agent.position)

                    # Makes a deep copy of current agent fitness to the best agent
                    space.best_agent.fit = copy.deepcopy(agent.fit)
    
    def _update(self, clans, function):
        
        centers = []
        
        for i in range(0, self.n_clans):
            clans[i] = sorted(clans[i], key=lambda x: x.fit)
            
            center = np.mean(np.array([agent.position for agent in clans[i]]), axis=0)
            
            centers.append(copy.deepcopy(center))
            
        # Clan updating operator
        for i, ci in enumerate(clans):
            # Iterates through all agents
            for j, agent in enumerate(ci):
                # Creates a temporary agent
                a = copy.deepcopy(agent)
                
                # Generating an uniform random number
                r1 = r.generate_uniform_random_number()
                
                if j == 0:
                    # Updating position according to Eq. 2
                    a.position = self.beta * centers[i]
                else:
                    # Updating position according to Eq. 1
                    a.position = agent.position + self.alpha * (ci[0].position - agent.position) * r1
                
                # Checking agents limits
                a.clip_limits()
                
                # Evaluates the agent
                a.fit = function(a.position)

                # If the new potision is better than the current agent's position
                if a.fit < agent.fit:
                    # Replace the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replace its fitness
                    agent.fit = copy.deepcopy(a.fit)
                    
        # Separating operator
        for i, ci in enumerate(clans):
            # Sorting agents
            clans[i] = sorted(clans[i], key=lambda x: x.fit)
            
            # Generate a new agent
            for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                    # For each decision variable, we generate uniform random numbers
                    ci[-1].position[j] = r.generate_uniform_random_number(lb, ub, size=agent.n_dimensions)
                    
            # Checking agents limits
            ci[-1].clip_limits()
    
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
        
        clans = self._create_clans(space.agents)
        
        # Initial search space evaluation
        self._evaluate(space, clans, function, hook=pre_evaluation)
        
        # We will define a History object for further dumping
        history = h.History(store_best_only)
        
        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')
                
                # Updating agents
                self._update(clans, function)

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, clans, function, hook=pre_evaluation)
                
                 # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')
                
        return history
