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


class ASO(Optimizer):
    """An ASO class, inherited from Optimizer.
    This is the designed class to define ASO-related
    variables and methods.
    References:
        W. Zhao, L. Wang and Z. Zhang.
        A novel atom search optimization for dispersion coefficient estimation in groundwater.
        Future Generation Computer Systems (2019).
    """

    def __init__(self, algorithm='ASO', hyperparams=None):
        """Initialization method.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> ASO.')

        # Override its parent class with the receiving hyperparams
        super(ASO, self).__init__(algorithm)
        
        # Depth weight
        self.alpha = 50

        # Multiplier weight
        self.beta = 0.2

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def alpha(self):
        """float: depth weight.
        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')

        self._alpha = alpha

    @property
    def beta(self):
        """float: multiplier weight.
        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0 or beta > 1:
            raise e.ValueError('`beta` should be between 0 and 1')

        self._beta = beta
        
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

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s| Hyperparameters: alpha = %s, beta = %s | '
                     'Built: %s.',
                     self.algorithm, self.alpha, self.beta, self.built)
    
    def _calculate_mass(self, agents):
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        sum_fit = np.sum([np.exp((agent.fit - agents[0].fit)/(agents[-1].fit - agents[0].fit) ) for agent in agents])
        
        mass = np.zeros((len(agents)))
        
        for i, agent in enumerate(agents):
            
            mass[i] = np.exp((agent.fit - agents[0].fit)/(agents[-1].fit - agents[0].fit) ) / sum_fit
        
        return mass
    
    def _calculate_acceleration(self, agents, best_agent, mass, t, n_iterations):
        
        acceleration = np.zeros((len(agents), best_agent.n_variables, best_agent.n_dimensions))
        
        G = np.exp(-20.0 * t / n_iterations)

        k_best = int(len(agents) - (len(agents) - 2) * ((t + 1) / n_iterations) ** 0.5) + 1

        k_best_agents, _ = map(list, zip(*sorted(zip(agents, mass), key=lambda x: x[1], reverse=True)[:k_best]))

        mk_average = np.mean([agent.position for agent in k_best_agents])
        
        for i, agent in enumerate(agents):
            
            average_dist = np.linalg.norm(agent.position - mk_average)

            summation = np.zeros((agent.n_variables, agent.n_dimensions))
            
            for atom in k_best_agents:
                
                radius = np.linalg.norm(agent.position - atom.position)
                
                rsmin = 1.1 + 0.1 * np.sin((t+1) / n_iterations * np.pi / 2)
                
                rsmax = 1.24
                
                if radius/average_dist < rsmin:
                    rs = rsmin
                else:
                    if radius/average_dist > rsmax:
                        rs = rsmax
                    else:
                        rs = radius / average_dist
                
                # Generating an uniform random number
                rand = r.generate_uniform_random_number()
                
                summation += (1 - t / n_iterations) ** 3 * (12 * (-rs)**(-13) - 6 * (-rs)**(-7)) * rand * ((atom.position - agent.position)/(radius + c.EPSILON))
                
            acceleration[i] = G * self.alpha * summation + self.beta * (best_agent.position - agent.position) / mass[i]

        return acceleration
    
    def _update_velocity(self, velocity, acceleration):
        """Updates an atom velocity (Eq. 30).
        Args:
            velocity (np.array): Agent's current velocity.
            acceleration (np.array): Agent's current acceleration.
        Returns:
            A new velocity.
        """

        # Generating first random number
        r1 = r.generate_uniform_random_number()

        # Calculates new velocity
        new_velocity = r1 * (velocity + acceleration)

        return new_velocity
    
    def _update_position(self, position, velocity):
        """Updates an atom position (Eq. 31).
        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.
        Returns:
            A new position.
        """

        # Calculates new position
        new_position = position + velocity

        return new_position
    
    def _update(self, agents, best_agent, function, velocity, t, n_iterations):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            velocity (np.array): Array of current velocities.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.
        """
        
        mass = self._calculate_mass(agents)
        
        acceleration = self._calculate_acceleration(agents, best_agent, mass, t, n_iterations)
        
        # Iterates through all agents
        for i, agent in enumerate(agents):
            
            # Updates current agent velocity
            velocity[i] = self._update_velocity(velocity[i], acceleration[i])

            # Updates current agent position
            agent.position = self._update_position(agent.position, velocity[i])
        
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
        
        velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent, function, velocity, t, space.n_iterations)

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
