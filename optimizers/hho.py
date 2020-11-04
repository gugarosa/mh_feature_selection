import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class HHO(Optimizer):
    """A HHO class, inherited from Optimizer.
    This is the designed class to define HHO-related
    variables and methods.
    References:
        A. A. Heidari and S. Mirjalili and H. Faris and I. Aljarah and M. Mafarja and H. Chen.
        Harris hawks optimization: Algorithm and applications. Future Generation Computer Systems (2019).
    """

    def __init__(self, algorithm='HHO', hyperparams=None):
        """Initialization method.
        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> HHO.')

        # Override its parent class with the receiving hyperparams
        super(HHO, self).__init__(algorithm)
        
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
    
    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.
        """
        
        # Iterates through all agents
        for agent in agents:
            # Generating an uniform random number
            rand = r.generate_uniform_random_number()
            
            # Updates the initial energy
            E0 = 2 * rand - 1
            
            # Updates the jump strength
            J = 2 * (1 - rand)
            
            # Updates the energy of a prey
            E = 2 * E0 * (1 - (iteration + 1) * 1.0 / n_iterations)
            
            # Exploration phase
            if (abs(E) >= 1):
                
                # Generating an uniform random number
                q = r.generate_uniform_random_number()
                
                if (q >= 0.5):
                    # Samples a random integer
                    rand = r.generate_integer_random_number(0, len(agents))
                    
                    # Mimics an agent position
                    X_rand = copy.deepcopy(agents[rand])
                    
                    # Generating an uniform random number
                    r1 = r.generate_uniform_random_number()
                    
                    # Generating an uniform random number
                    r2 = r.generate_uniform_random_number()
                    
                    # Updates the location vector using Eq. 1
                    agent.position = X_rand.position - r1 * abs(X_rand.position - 2 * r2 * agent.position)

                else:
                    # Averaged position of the Hawk population
                    averaged_position = np.mean([x.position for x in agents], axis=0)
                    
                    # Generating an uniform random number
                    r3 = r.generate_uniform_random_number()
                    
                    # Generating an uniform random number
                    r4 = r.generate_uniform_random_number()
                    
                    # Updates the location vector using Eq. 1
                    agent.position = (best_agent.position - averaged_position) - r3 * (np.expand_dims(agent.lb, -1) + r4 * (np.expand_dims(agent.ub, -1) - np.expand_dims(agent.lb, -1)))

            # Exploitation phase
            else:
                # Generating an uniform random number
                w = r.generate_uniform_random_number()
                
                # Soft besiege
                if (w >= 0.5 and abs(E) >= 0.5):
                    
                    delta_position = best_agent.position - agent.position
                    
                    # Updates the location vector using Eq. 4
                    agent.position = delta_position - E * abs(J * best_agent.position - agent.position)
                
                # Hard besiege
                elif (w >= 0.5 and abs(E) >= 0.5):
                    
                    delta_position = best_agent.position - agent.position
                    
                    # Updates the location vector using Eq. 6
                    agent.position = best_agent.position - E * abs(delta_position)
                
                # Soft besiege with progressive rapid dives
                elif (w < 0.5 and abs(E) >= 0.5):
                    
                    Y = best_agent.position - E * abs(J * best_agent.position - agent.position)
                    
                    # Generates a Lévy distribution
                    LF = d.generate_levy_distribution(1.5, (agent.n_variables, agent.n_dimensions))
                    
                    S = r.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))
                    
                    Z = Y + S * LF
                    
                    Y_fit = function(Y)
                    
                    Z_fit = function(Z)
                    
                    if (Y_fit < agent.fit):
                        # Updates the corresponding agent's position
                        agent.position = copy.deepcopy(Y)
                        
                        # And its fitness as well
                        agent.fit = copy.deepcopy(Y_fit)
                        
                    if (Z_fit < agent.fit):
                        # Updates the corresponding agent's position
                        agent.position = copy.deepcopy(Z)
                        
                        # And its fitness as well
                        agent.fit = copy.deepcopy(Z_fit)
                
                # Hard besiege with progressive rapid dives
                elif (w < 0.5 and abs(E) < 0.5):
                    
                    # Averaged position of the Hawk population
                    averaged_position = np.mean([x.position for x in agents], axis=0)
                    
                    Y = best_agent.position - E * abs(J * best_agent.position - averaged_position)
                    
                    # Generates a Lévy distribution
                    LF = d.generate_levy_distribution(1.5, (agent.n_variables, agent.n_dimensions))
                    
                    S = r.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))
                    
                    Z = Y + S * LF
                    
                    Y_fit = function(Y)
                    
                    Z_fit = function(Z)
                    
                    if (Y_fit < agent.fit):
                        # Updates the corresponding agent's position
                        agent.position = copy.deepcopy(Y)
                        
                        # And its fitness as well
                        agent.fit = copy.deepcopy(Y_fit)
                        
                    if (Z_fit < agent.fit):
                        # Updates the corresponding agent's position
                        agent.position = copy.deepcopy(Z)
                        
                        # And its fitness as well
                        agent.fit = copy.deepcopy(Z_fit)

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
                self._update(space.agents, space.best_agent, function, t, space.n_iterations)

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
