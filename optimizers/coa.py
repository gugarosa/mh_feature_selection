"""Coyote optimization algorithm.
"""

import copy
import math
import sys

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

import logging
logging.disable(sys.maxsize)


class COA(Optimizer):
    """A COA class, inherited from Optimizer.

    This is the designed class to define COA-related
    variables and methods.

    References:
        J. Pierezan, L.S. Coelho Coyote optimization algorithm: a new metaheuristic for global optimization problems.
        Proc. IEEE Congr. Evol. Comput. (2018), pp. 2633-2640.

    """

    def __init__(self, algorithm='COA', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> COA.')

        # Override its parent class with the receiving hyperparams
        super(COA, self).__init__(algorithm)
        
        # number of packs
        self.Np = 2
        
        # number of coyotes per pack
        self.Nc = 5       
        
        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
        
    @property
    def Np(self):
        """int: number of packs.

        """

        return self._Np

    @Np.setter
    def Np(self, Np):
        if not isinstance(Np, (float, int)):
            raise e.TypeError('`Np` should be a float or integer')
        if Np <= 1:
            raise e.ValueError('`Np` should be greater or equal to 1')
        self._Np = Np
        
    @property
    def Nc(self):
        """int: number of coyotes per pack.

        """

        return self._Nc

    @Nc.setter
    def Nc(self, Nc):
        if not isinstance(Nc, (float, int)):
            raise e.TypeError('`Nc` should be a float or integer')
        if Nc <= 1 or Nc>14:
            raise e.ValueError('`Nc` should be greater or equal to 1 and smallar than 15')
        self._Nc = Nc
              
       
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
            if 'Np' in hyperparams:
                self.Np = hyperparams['Np']
            if 'Nc' in hyperparams:
                self.Nc = hyperparams['Nc']                  
               

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Hyperparameters: Np = %s, Nc = %s| Built: %s.',
                     self.algorithm, self.Np, self.Nc, self.built)
                     
    def _detectAlpha(self, pack, agents):
        """This method is employed to detect the alpha coyote of a pack, i.e., the best individual from the pack.

        Args:
            pack (List): A list of indexes from individuals compounding the pack.
            agents (list): List of agents.
        """
        
        min_fit = float('inf') 
        best = -1
        for c in range(len(pack)):
            if agents[pack[c]].fit < min_fit:
                best = c
                min_fit = agents[pack[c]].fit
        return best
            
    def _tendency(self, pack, agents):
        """This method is employed to detect the cultural tendency of the pack.

        Args:
            pack (List): A list of indexes from individuals compounding the pack.
            agents (list): List of agents.
        """
        
        tendency = np.zeros((agents[0].n_variables, agents[0].n_dimensions)) 
        
        fits = np.zeros(len(pack))
        for c in range(len(pack)):
            fits[c] = agents[pack[c]].fit     

        pack_ordered = pack[np.argsort(fits)]

        if len(pack) %2!=0:
            for v in range(len(tendency)):  
                index_odd =  pack_ordered[int((len(pack_ordered)+1)/2) -1]    
                tendency[v,:] = copy.deepcopy(agents[index_odd].position[v,:])
        else:
            for v in range(len(tendency)):    
                index_even_1 = pack_ordered[int((len(pack_ordered))/2) -1]
                index_even_2 = pack_ordered[int((len(pack_ordered))/2)]
                tendency[v,:] = (agents[index_even_1].position[v,:] + agents[index_even_2].position[v,:])/2
        return tendency

    def _computeDeltas(self, pack, agents, alpha, tendency):
        """Method that computes the influence of alpha ( delta_1 ) and by other coyotes of the pack ( delta_2).
        Args:

            pack (List): A list of indexes from individuals compounding the pack.
            agents (list): List of agents.

            alpha (int): Index from best agent in the pack.
            tendency (agent): Average social condition of the pack.
        """
        delta_1 = np.zeros((agents[0].n_variables, agents[0].n_dimensions)) 
        delta_2 = np.zeros((agents[0].n_variables, agents[0].n_dimensions)) 

        while True:
            cr1 = r.generate_integer_random_number(0, len(pack))
            if cr1 != alpha and len(pack)>0:
                break

        delta_1 = np.subtract(agents[pack[alpha]].position, agents[pack[cr1]].position)

        cr2 = r.generate_integer_random_number(0, len(pack))      
        delta_2 = np.subtract(tendency, agents[pack[cr2]].position)

        return delta_1, delta_2


    def _updateSocialCondition(self, pack, space, delta_1, delta_2, function):
        """Method that updates the social condition from coyotes in a pack.
        Args:

            pack (List): A list of indexes from individuals compounding the pack.
            space (SearchSpace): Search space.

            delta_1 (list): Alpha's influence.
            delta_2 (list): Other coyotes' influence.
        """

        agents = space.agents
        # weights of the alpha influence
        r1 = r.generate_uniform_random_number()

        # weights of the pack influence
        r2 = r.generate_uniform_random_number()

        new_agente = Agent(agents[0].n_variables, agents[0].n_dimensions)  
        # For all coyotes in pack p
        for c in range(len(pack)):
            new_agente.position = agents[pack[c]].position + r1*delta_1 + r2*delta_2     #np.add(agents[pack[c]].position, np.add(r1*delta_1 , r2*delta_2))
            new_agente.clip_limits()

            new_agente.fit = function(new_agente.position)
            if (new_agente.fit < agents[pack[c]].fit):
                # update the social condition of the coyote */
                for k in range(agents[0].n_variables):
                    agents[pack[c]].position[k] = copy.deepcopy( new_agente.position[k])
                agents[pack[c]].fit = copy.deepcopy(new_agente.fit) 

    def _generateNewPuppet(self,pack, agents, function):
        """Birth of a new coyote.
        Args:

            pack (List): A list of indexes from individuals compounding the pack.
            agents (list): List of agents.
        """
        puppet = Agent(agents[0].n_variables, agents[0].n_dimensions)

        # Select two random coyotes from the pack
        while True:
            r1 = r.generate_integer_random_number(0, len(pack))
            r2 = r.generate_integer_random_number(0, len(pack))
            if r1 != r2:
                break

        # Select two random features 
        while True:
            j1 = r.generate_integer_random_number(0, agents[0].n_variables)
            j2 = r.generate_integer_random_number(0, agents[0].n_variables)
            if j1 != j2:
                break
        # scatter probability 
        Ps = 1/agents[0].n_variables

        # association probability 
        Pa = 1-Ps/2
        for j in range(agents[0].n_variables):
            rnd_j = r.generate_uniform_random_number()
            if rnd_j<Ps or j==j1:
                puppet.position[j,:] = copy.deepcopy(agents[pack[r1]].position[j,:])  
            elif rnd_j>=Ps+Pa or j==j2:
                puppet.position[j,:] = copy.deepcopy(agents[pack[r2]].position[j,:])  
            else:
                Rj = np.zeros(agents[0].n_dimensions)
                for d in range(agents[0].n_dimensions):
                    Rj[d] = r.generate_uniform_random_number(agents[0].lb[j],agents[0].ub[j])
                puppet.position[j,:] = copy.deepcopy(Rj)  
        puppet.clip_limits()

        puppet.fit = function(puppet.position)
        return puppet

    def _computeBirthAndDeath(self,pack, space, function, coyotes_age):
        """Method that computes birth of new coyotes based on two parents and the environmental influence, as well as the death of a coyote.
        Args:

            pack (List): A list of indexes from individuals compounding the pack.
            agents (list): List of agents.
        """

        agents = space.agents

        puppet = self._generateNewPuppet(pack, agents, function)
        # Find the set of coyotes with worst adaptation regarding the puppet (omega)
        omega = []
        for c in range(len(pack)):
            if puppet.fit < agents[pack[c]].fit:
                
                values = np.array([c, agents[pack[c]].fit, coyotes_age[pack[c]]] )
                omega.append( values   )
        omega = np.asarray(omega)

        # Find the number of coyotes with worst adaptation regarding the puppet in the pack
        psi = len(omega)

        if psi==1:
            # if there is only one coyote worst than the puppet, then this coyote dies and the puppet takes his place
            agents[pack[int(omega[0,0])]].position = copy.deepcopy(puppet.position)
            agents[pack[int(omega[0,0])]].fit = copy.deepcopy(puppet.fit)
            coyotes_age[pack[int(omega[0,0])]] = 0
            
        elif psi>1:
            # if there is more one coyote worst than the puppet, then the oldest dies and the puppet takes his place
            max_age = np.max(omega[:,2])
            index_max = np.where(omega[:,2] == max_age)[0]

            if len(index_max)==1:
                agents[pack[int(omega[index_max[0],0])]].fit = copy.deepcopy(puppet.fit)

                # sets the age from newborn coyote
                coyotes_age[pack[int(omega[index_max[0],0])]] = 0

            else:
                # if there is more one with the same age, then the worst dies and the puppet takes his place
                max_fit = np.max(omega[index_max,1])
                index_max_fit = np.where(omega[index_max,1] == max_fit)[0]
                agents[pack[int(omega[index_max_fit[0],0])]].fit = copy.deepcopy(puppet.fit)

                # sets the age from newborn coyote
                coyotes_age[pack[int(omega[index_max_fit[0],0])]] = 0
        else:
            pass
            # puppet dies

    def _transition(self, packs):
        """Computes the eviction probability and transition to another pack.
        Args:
            packs (list of np.arrays): An Np x Nc where each row describes a pack and each col is the index of an individual from the pack.
        """
        # eviction probability 
        Pe = 0.005 * (self.Nc * self.Nc)
        
        if r.generate_uniform_random_number() < Pe and len(packs)>1:
            while True:
                p1 = r.generate_integer_random_number(0, len(packs))
                p2 = r.generate_integer_random_number(0, len(packs))
                if p1 != p2:
                    break

            c1 = r.generate_integer_random_number(0, len(packs[p1]))
            c2 = r.generate_integer_random_number(0, len(packs[p2]))

            aux = packs[p1,c1]
            packs[p1,c1] = packs[p2,c2]
            packs[p2,c2] = aux

    def _update(self, space, function, packs, coyotes_age):
        """Method that updates Coyotes' social organization and their adaptation to environment.
        Args:
            space (SearchSpace): Search space.
            function (Function): A Function object that will be used as the objective function.
            packs (list of np.arrays): An Np x Nc where each row describes a pack and each col is the index of an individual from the pack.
        """
        agents = space.agents
        alpha = np.zeros(len(packs)).astype(int)
        tendency = np.zeros((len(packs),agents[0].n_variables, agents[0].n_dimensions)) 
        
        # Iterate through all packs
        for p in range(len(packs)):
            # Detect index of the alpha coyote from pack p
            alpha[p] = self._detectAlpha(packs[p], agents)
            # Compute the cultural tendency of the pack p
            tendency[p] = self._tendency(packs[p], agents)

            # compute deltas to update the social condition
            delta1, delta2 = self._computeDeltas(packs[p], agents, alpha[p], tendency[p] )


            # Update the social condition	
            self._updateSocialCondition(packs[p], space, delta1, delta2, function )

            # Compute birth and death of a coyote
            self._computeBirthAndDeath(packs[p], space, function, coyotes_age)

        # Transition between packs
        self._transition(packs)

        # Update coyote's age
        coyotes_age = coyotes_age + 1

        for a in range(space.n_agents):
            if agents[a].fit < space.best_agent.fit:
                # It updates the global best value and position 
                space.best_agent.position = copy.deepcopy(agents[a].position) 
                space.best_agent.fit = copy.deepcopy(agents[a].fit) 

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


        if self.Nc * self.Np != space.n_agents:
            raise e.ValueError('The total number of coyotes (agenst) should be equal to the number of packs times the number of coyotes per pack.')        
        
        # create the packs

        rng = np.random.default_rng()
        packs = np.arange(space.n_agents)
        rng.shuffle(packs)

        coyotes_age = np.ones((space.n_agents)).astype(int)
        
        packs = np.reshape(packs,(self.Np, self.Nc)) 

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space, function, packs, coyotes_age)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history
