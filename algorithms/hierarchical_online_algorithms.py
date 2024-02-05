import numpy as np
import sys 

sys.path.append("..")

from utils.subtasks import learn_subtasks
from abc import ABC

class HierarchicalAlgorithm(ABC):

    def __init__(self, env, lrs, eta):

        self.env = env 
        self.lrs = lrs 
        self.eta = eta

        self.subtasks = HierarchicalAlgorithm._prepare_subtasks(env)


    def sample_next_state(self, state):

        state_idx = self.env.states.index(state)
        next_states_idxs = np.where(self.env.T[state_idx, :] > 0)[0].tolist()
        next_states = [self.env.states[i] for i in next_states_idxs]

        next_states_values = np.fromiter(map(self.get_composed_value, next_states), dtype=np.float64)

        pi = next_states_values / np.sum(next_states_values)

        next_state_idx = np.random.choice(next_states_idxs, p=pi)
        next_state = self.env.states[next_state_idx]

        return next_state, (1/len(next_states)) / pi[next_states.index(next_state)]
    
    
    def get_composed_value(self, state):

        if state in self.env.TS:
        
            return self.exit_estimates[self.env.exit_states.index(state)]
        
        else:
            
            partition = self.env.get_partition(state)
            abs_state = self.env.project_state_in_partition(state, partition)
            exit_states_idxs = [self.env.exit_states.index(e) for e in self.env.partitions[partition]['exit_states']]
            w = self.exit_estimates[exit_states_idxs]
            z = np.dot(w, self.subtasks)

            return z[self.env.abs_states.index(abs_state)]
        
    
    def update_subtasks(self, **args):
        
        abs_state = args["abs_state"]
        update = args["update"]

        if update:
            
            abs_s_idx = self.env.abs_states.index(abs_state)
            r = -self.env.abs_R[:, abs_s_idx] - self.rho

            next_abs_idxs = np.where(self.env.abs_T[abs_s_idx, :])[0].tolist()
            target = np.exp(r) * np.nanmean(self.subtasks[:, next_abs_idxs], axis=1)
            
            self.subtasks[:, self.env.abs_states.index(abs_state)] += self.lrs.lr2 * \
                (target - self.subtasks[:, self.env.abs_states.index(abs_state)])
        
    
    def update_lrs(self):

        self.lrs.update()

    @staticmethod   
    def _prepare_subtasks(env):
        subtasks, _ = learn_subtasks(env.problem_id, 0, env.DIM, 1)
        subtasks[np.where(subtasks > 0)] = 1
        return subtasks

        
    @property
    def exit_states_idxs(self):
        return list(map(self.env.states.index, self.env.exit_states))

        

class HierarchicalExp(HierarchicalAlgorithm):


    def __init__(self, env, lrs, eta=1):

        super().__init__(env, lrs, eta)

        self.gamma = 0.5

        self.exit_estimates = HierarchicalExp._prepare_exit_estimates(env)
    
    
    def update(self, **args):
        
        state = args["state"]
        next_state = args["next_state"]
        r = args["reward"]
        w = args["isw"]
        update_vf = args["update_vf"]

        state_value = self.get_composed_value(state)
        next_state_value = self.get_composed_value(next_state)


        deltaG = (
                w * (np.exp(-r) * next_state_value) / state_value - self.gamma)
        
        self.gamma += self.lrs.lr3 * deltaG

        if state in self.env.exit_states and update_vf:

            e_idx = self.env.exit_states.index(state)
            deltaZ = (self.get_composed_value(state) - self.exit_estimates[e_idx])
            self.exit_estimates[e_idx] += self.lrs.lr1 * deltaZ


    @staticmethod
    def _prepare_exit_estimates(env):
        exit_states_idxs = list(map(env.states.index, env.exit_states))
        exit_estimates = np.exp(-env.R[exit_states_idxs])
        exit_estimates[np.where(exit_estimates > 0)] = 1
        return exit_estimates
    
    @property
    def rho(self):
        return np.log(self.gamma)

    def get_policy(self, state):

        pass


class HierarchicalLog(HierarchicalAlgorithm):


    def __init__(self, env, lrs, eta=1):

        super().__init__(env, lrs, eta)

        self.rho = np.log(0.3468270998021556)

        self.exit_estimates_vf = HierarchicalLog._prepare_exit_estimates(env)

    def sample_next_state(self, state):

        state_idx = self.env.states.index(state)
        next_states_idxs = np.where(self.env.T[state_idx, :] > 0)[0].tolist()
        next_states = [self.env.states[i] for i in next_states_idxs]

        next_states_values = np.fromiter(map(self.get_composed_value, next_states), dtype=np.float64)

        pi = next_states_values / np.sum(next_states_values)

        next_state_idx = np.random.choice(next_states_idxs, p=pi)
        next_state = self.env.states[next_state_idx]

        return next_state, (1/len(next_states)) / pi[next_states.index(next_state)]
    
    
    def update(self, **args):
        
        state = args["state"]
        r = args["reward"]
        w = args["isw"]
        next_state = args["next_state"]
        update_vf = args["update_vf"]

        # TODO: Review this

        next_states_idxs = np.where(self.env.T[self.env.states.index(state), :] > 0)[0].tolist()

        next_states_values = [self.get_composed_value(self.env.states[ns]) for ns in next_states_idxs]

        # print(state, next_state, next_states_values)

        delta = (-r + (- self.rho + np.log(np.dot(self.env.T[self.env.states.index(state), next_states_idxs], next_states_values)) \
                       - self.get_composed_value(state)) / self.eta)
        
        self.rho += self.lrs.lr3 * delta

        if state in self.env.exit_states and update_vf:

            e_idx = self.env.exit_states.index(state)
            deltaZ = np.log(self.get_composed_value(state)) - self.exit_estimates_vf[e_idx]
            self.exit_estimates_vf[e_idx] += self.lrs.lr1 * deltaZ


    def get_composed_value(self, state):

        if state in self.env.TS:
        
            return self.exit_estimates[self.env.exit_states.index(state)]
        
        else:
            
            partition = self.env.get_partition(state)
            abs_state = self.env.project_state_in_partition(state, partition)
            exit_states_idxs = [self.env.exit_states.index(e) for e in self.env.partitions[partition]['exit_states']]
            w = self.exit_estimates[exit_states_idxs]
            z = np.dot(w, self.subtasks)

            return z[self.env.abs_states.index(abs_state)]
        
    
    def update_lrs(self):

        self.lrs.update()

    @staticmethod
    def _prepare_exit_estimates(env):
        exit_states_idxs = list(map(env.states.index, env.exit_states))
        exit_estimates = -env.R[exit_states_idxs]
        exit_estimates[np.where(np.exp(exit_estimates) > 0)] = -1
        return exit_estimates
    
    @property
    def exit_estimates(self):
        return np.exp(self.exit_estimates_vf)


    @property
    def exit_states_idxs(self):
        return list(map(self.env.states.index, self.env.exit_states))

    @property
    def gamma(self):
        return np.exp(self.rho)
    
    def get_policy(self, state):

        pass