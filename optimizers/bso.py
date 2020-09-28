"""Brain Storm Optimization.
"""

import copy
import sys
import math


import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BSO(Optimizer):
    """A BSO class, inherited from Optimizer.

    This is the designed class to define BSO-related
    variables and methods.

    References:
        Y. Shi. Brain Storm Optimization Algorithm.
        International Conference in Swarm Intelligence (2011).

    """

    def __init__(self, algorithm='BSO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BSO.')

        # Override its parent class with the receiving hyperparams
        super(BSO, self).__init__(algorithm)
        
        # number of clusters
        self.k = 3
        
        # probability of selecting a cluster center
        self.p_one_cluster = 0.3
        
        # probability of randomly selecting an idea from a probabilistic selected cluster
        self.p_one_center = 0.4
        
        # probability of creating a random combination of two probabilistic selected clusters
        self.p_two_centers = 0.3

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def k(self):
        """int: number of clusters.

        """

        return self._k

    @k.setter
    def k(self, k):
        if not isinstance(k, (float, int)):
            raise e.TypeError('`k` should be a float or integer')
        if k <= 1:
            raise e.ValueError('`k` should be greater or equal to 1')
        self._k = k
        
    @property
    def p_one_cluster(self):
        """float: probability of selecting a cluster center.

        """

        return self._p_one_cluster

    @p_one_cluster.setter
    def p_one_cluster(self, p_one_cluster):
        if not isinstance(p_one_cluster, (float, int)):
            raise e.TypeError('`p_one_cluster` should be a float or integer')
        if p_one_cluster < 0 or p_one_cluster > 1:
            raise e.ValueError('`p_one_cluster` should be between 0 and 1')
        self._p_one_cluster = p_one_cluster
        
    @property
    def p_one_center(self):
        """float: probability of randomly selecting an idea from a probabilistic selected cluster.

        """

        return self._p_one_center

    @p_one_center.setter
    def p_one_center(self, p_one_center):
        if not isinstance(p_one_center, (float, int)):
            raise e.TypeError('`p_one_center` should be a float or integer')
        if p_one_center < 0 or p_one_center > 1:
            raise e.ValueError('`p_one_center` should be between 0 and 1')
        self._p_one_center = p_one_center
        
    @property
    def p_two_centers(self):
        """float: probability of creating a random combination of two probabilistic selected clusters

        """

        return self._p_two_centers

    @p_two_centers.setter
    def p_two_centers(self, p_two_centers):
        if not isinstance(p_two_centers, (float, int)):
            raise e.TypeError('`p_two_centers` should be a float or integer')
        if p_two_centers < 0 or p_two_centers > 1:
            raise e.ValueError('`p_two_centers` should be between 0 and 1')
        self._p_two_centers = p_two_centers
        
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
            if 'k' in hyperparams:
                self.k = hyperparams['k']
            if 'p_one_cluster' in hyperparams:
                self.p_one_cluster = hyperparams['p_one_cluster']
            if 'p_one_center' in hyperparams:
                self.p_one_center = hyperparams['p_one_center']
            if 'p_two_centers' in hyperparams:
                self.p_two_centers = hyperparams['p_two_centers']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Hyperparameters: k = %s, p_one_cluster = %s, p_one_center = %s, p_two_centers = %s| Built: %s.',
                     self.algorithm, self.k, self.p_one_cluster, self.p_one_center, self.p_two_centers, self.built)
        
        
        
    def _runKmeans(self, space):
        """Method that computes kmeans to cluster best solutions.
        Args:
            space (Searchspace): Current search space.

        Returns:
            ideas_per_cluster (List of nparrays): set of agents per cluster.
            best_fitness_cluster (List): List of best finess value per cluster.
            best_ideas (List): List of idea (agent) per cluster.
        """

        # randomly select the agents that will be the cluster centers
        # -- change for higher dimensional space
        #center = np.zeros((self.k, space.n_variables, space.n_dimensions))
        center = np.zeros((self.k, space.n_variables))
        randnums= np.random.randint(0,space.n_agents,self.k)
        for i in range(self.k):      
            # -- change for higher dimensional space      
            #center[i] = space.agents[randnums[i]].position
            center[i] = space.agents[randnums[i]].position[:,0]

        # -- change for higher dimensional space
        #X = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        X = np.zeros((space.n_agents, space.n_variables))
        for i in range(space.n_agents):
            X[i] = space.agents[i].position[:,0]
        # clustering ideas
        clusters_labels = KMeans(n_clusters=self.k, init=center, n_init=1).fit(X).labels_

        ideas_per_cluster = []
        best_fitness_cluster = np.ones(self.k)*sys.float_info.max
        best_ideas = np.zeros(self.k)
        for i in range(self.k):
            agents_cluster_i = np.asarray(np.where(clusters_labels == i))[0]
            ideas_per_cluster.append(agents_cluster_i)
            for j in agents_cluster_i: 
                if best_fitness_cluster[i]>space.agents[j].fit:
                    best_fitness_cluster[i] = space.agents[j].fit
                    best_ideas[i] = j
                    	
        ideas_per_cluster = ideas_per_cluster
        return ideas_per_cluster, best_fitness_cluster, best_ideas.astype(int)

    def _sigmoid(self, x):
      return 1 / (1 + math.exp(-x))
        
    def _update(self, space, function, t):
        """Method that updates the brainstorm solutions.
        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            t (int): Current iteration.
        """
        ideas_per_cluster, best_fitness_cluster, best_ideas = self._runKmeans(space)

        nidea = Agent(space.n_variables, space.n_dimensions)      

        # Iterate through all agents
        for i, agent in enumerate(space.agents):
            p = r.generate_uniform_random_number()
            if self.p_one_cluster > p:
                # selecting a cluster probabilistically
                c1 = r.generate_integer_random_number(0, self.k) 
                p = r.generate_uniform_random_number()
                # creating a new idea based on the cluster selected previously.
                #   We also consider if cluster c1 has a single idea, i.e., len(ideas_per_cluster[c1]) == 0.
                #   Notice we do not consider the cluster's center into that computation @kmeans function, which means a
                #   unitary cluster has len(ideas_per_cluster[c1]) == 0.

                if (self.p_one_center > p) or (len(ideas_per_cluster[c1])  == 0):
                    for k in range(space.n_variables):
                        nidea.position[k,:] = copy.deepcopy(space.agents[ best_ideas[c1]].position[k,:])	
                else:
                    # creating a new idea based on another idea j selected randomly from cluster c1
                    j = int(r.generate_integer_random_number(0, len(ideas_per_cluster[c1])))
                    j = ideas_per_cluster[c1][j]

                    for k in range(space.n_variables):
                        nidea.position[k,:] = copy.deepcopy(space.agents[j].position[k,:])
            else:
                # selecting two clusters' centers probabilistically 
                while True:
                    c1 = int(r.generate_integer_random_number(0, self.k))
                    c2 = int(r.generate_integer_random_number(0, self.k))
                    if c1 != c2 or self.k ==1:
                        break

                # selecting two ideas randomly 
                if len(ideas_per_cluster[c1]) == 0:
                    j = best_ideas[c1]
                else:
                    j = r.generate_integer_random_number(0, len(ideas_per_cluster[c1]))
                    j = ideas_per_cluster[c1][j]

                if len(ideas_per_cluster[c2]) == 0:
                    z = best_ideas[c2]
                else:
                    z = r.generate_integer_random_number(0, len(ideas_per_cluster[c2]));
                    z = ideas_per_cluster[c2][z];

                p = r.generate_uniform_random_number()
                rand = r.generate_uniform_random_number()

                # it creates a new idea based on a random combination of two selected clusters' centers 
                if self.p_two_centers > p:
                    for k in range(space.n_variables):
                        nidea.position[k,:] = (rand* space.agents[ best_ideas[c1]].position[k,:]) + ((1 - rand)* space.agents[ best_ideas[c2]].position[k,:]) 
                else:
                    # it creates a new idea based on the ideas selected at random from the clusters previously chosen 
                    for k in range(space.n_variables):
                        nidea.position[k,:] = (rand* space.agents[ best_ideas[c1]].position[k,:]) + ((1 - rand)* space.agents[ best_ideas[c2]].position[k,:]) 
                        nidea.position[k,:] = (rand * space.agents[j].position[k,:]) + ((1 - rand) * space.agents[z].position[k,:])



				
            # adding local noise to the new created idea */
            p = (0.5 * space.n_iterations- t) / space.n_variables
            rand = r.generate_uniform_random_number() * self._sigmoid(p);

            for k in range(space.n_variables):
                nidea.position[k,:] += rand * r.generate_uniform_random_number(0, 1)

            # It evaluates the new created idea
            nidea.clip_limits()

            nidea.fit = function(nidea.position)
            if (nidea.fit < space.agents[i].fit):
                # if the new idea is better than the current one */
                for k in range(space.n_variables):
                    space.agents[i].position[k] = copy.deepcopy( nidea.position[k])
                space.agents[i].fit = copy.deepcopy(nidea.fit)

            if space.agents[i].fit < space.best_agent.fit:
                # It updates the global best value and position 
                space.best_agent.position = copy.deepcopy(space.agents[i].position)  
                space.best_agent.fit = copy.deepcopy(space.agents[i].fit)    
        
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
                self._update(space, function, t)

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
