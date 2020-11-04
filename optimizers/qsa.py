import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class QSA(Optimizer):
    """A QSA class, inherited from Optimizer.
    This is the designed class to define QSA-related
    variables and methods.
    References:
        Jinhao Zhang and Mi Xiao and Liang Gao and Quanke Pan.
        Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems. Applied Mathematical Modelling (2018).
    """
    
    def __init__(self, algorithm='QSA', hyperparams=None):
        """Initialization method.
        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logger.info('Overriding class: Optimizer -> QSA.')

        # Override its parent class with the receiving hyperparams
        super(QSA, self).__init__(algorithm)
        
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
    
    def _calculate_queue(self, n_agents, t1, t2, t3):
        
        if t1 > 1.0e-6:
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
            n3 = (1 / t3) / ((1 / t1) + (1 / t2) + (1 / t3))
        else:
            n1 = 1.0 / 3
            n2 = 1.0 / 3
            n3 = 1.0 / 3
        q1 = int(n1 * n_agents)
        q2 = int(n2 * n_agents)
        q3 = int(n3 * n_agents)
        
        return q1, q2, q3
    
    def _business_one(self, agents, function, beta):
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        A1, A2, A3 = copy.deepcopy(agents[0]), copy.deepcopy(agents[1]), copy.deepcopy(agents[2])
        
        q1, q2, q3 = self._calculate_queue(len(agents), A1.fit, A2.fit, A3.fit)
        
        # Represents the update patterns by Eq. 4 and Eq. 5
        case = None
        
        # Iterates through all agents
        for i, agent in enumerate(agents):
            
            # Creates a temporary agent
            a = copy.deepcopy(agent)
            
            if i < q1:
                if i == 0:
                    case = 1
                A = copy.deepcopy(A1)
            elif q1 <= i < q1 + q2:
                if i == q1:
                    case = 1
                A = copy.deepcopy(A2)
            else:
                if i == q1 + q2:
                    case = 1
                A = copy.deepcopy(A3)
            
            # Generating an uniform random number
            alpha = r.generate_uniform_random_number(-1, 1)
            
            # Generating a Erlang distribution
            E = np.random.gamma(1, 0.5, (agent.n_variables, agent.n_dimensions))
            
            if case == 1:
                F1 = beta * alpha * (E * np.abs(A.position - agent.position)) + np.random.gamma(1, 0.5, 1)[0] * (A.position - agent.position)
                
                # Obtain X_new based on Eq. 4
                a.position = A.position + F1
                
                # Evaluates the agent
                a.fit = function(a.position)

                # If X_new is better than the current agent
                if a.fit < agent.fit:
                    # Replace the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replace its fitness
                    agent.fit = copy.deepcopy(a.fit)
                    
                    case = 1
                else:
                    case = 2
            else:
                F2 = beta * alpha * (E * np.abs(A.position - agent.position))
                
                # Obtain X_new based on Eq. 5
                a.position = agent.position + F2
                
                # Evaluates the agent
                a.fit = function(a.position)

                # If X_new is better than the current agent
                if a.fit < agent.fit:
                    # Replace the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replace its fitness
                    agent.fit = copy.deepcopy(a.fit)
                    
                    case = 2
                else:
                    case = 1
    
    def _business_two(self, agents, function):
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        A1, A2, A3 = copy.deepcopy(agents[0]), copy.deepcopy(agents[1]), copy.deepcopy(agents[2])
        
        q1, q2, q3 = self._calculate_queue(len(agents), A1.fit, A2.fit, A3.fit)
        
        pr = [i / len(agents) for i in range(1, len(agents) + 1)]
        
        # Calculating the confusion degree
        cv = A1.fit / (A2.fit + A3.fit)
        
        # Iterates through all agents
        for i, agent in enumerate(agents):
            
            # Creates a temporary agent
            a = copy.deepcopy(agent)
            
            if i < q1:
                A = copy.deepcopy(A1)
            elif q1 <= i < q1 + q2:
                A = copy.deepcopy(A2)
            else:
                A = copy.deepcopy(A3)
            
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()
            
            if r1 < pr[i]:
                a1, a2 = np.random.choice(agents, 2, replace=False)
                
                # Generating an uniform random number
                r2 = r.generate_uniform_random_number()
                
                if r2 < cv:
                    # Obtain X_new based on Eq. 12
                    a.position = agent.position + np.random.gamma(1, 0.5, 1)[0] * (a1.position - a2.position)
                else:
                    # Obtain X_new based on Eq. 13
                    a.position = agent.position + np.random.gamma(1, 0.5, 1)[0] * (A.position - a1.position)
                    
                # Evaluates the agent
                a.fit = function(a.position)

                # If the new potision is better than the current agent's position
                if a.fit < agent.fit:
                    # Replace the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replace its fitness
                    agent.fit = copy.deepcopy(a.fit)
    
    def _business_three(self, agents, function):
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        pr = [i / len(agents) for i in range(1, len(agents) + 1)]
        
        # Iterates through all agents
        for i, agent in enumerate(agents):
            
            a = copy.deepcopy(agent)
            
            for j in range(agent.n_variables):
            
                # Generating an uniform random number
                r1 = r.generate_uniform_random_number()

                if r1 < pr[i]:
                    a1, a2 = np.random.choice(agents, 2, replace=False)
                    
                    # Obtain X_new based on Eq. 17
                    a.position[j] = a1.position[j] + np.random.gamma(1, 0.5, 1)[0] * (a2.position[j] - agent.position[j])
                
                # Evaluates the agent
                a.fit = function(a.position)

                # If X_new is better than the current agent
                if a.fit < agent.fit:
                    # Replace the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replace its fitness
                    agent.fit = copy.deepcopy(a.fit)
    
    def _update(self, agents, best_agent, function, beta):
        """Method that wraps the update pipeline over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
        """
    
        self._business_one(agents, function, beta)
        
        self._business_two(agents, function)
        
        self._business_three(agents, function)
    
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
                
                beta = np.power(t, np.power(t / n_iterations, 0.5))

                # Updating agents
                self._update(space.agents, space.best_agent, function, beta)

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
