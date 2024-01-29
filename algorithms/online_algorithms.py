from collections import defaultdict
import numpy as np
from collections import deque

class AbstractAlgorithm:

    def __init__(self, 
                 env,
                 decay_step_gamma: int,
                 decay_step_vf: int,
                 decay_factor_gamma: float,
                 decay_factor_vf: float,
                 init_lr_vf: int = 1,
                 init_lr_g: int = 1):

        self.env = env
        self.samples = 0
        self.lr_vf = init_lr_vf
        self.lr_g = init_lr_g
        self.decay_step_gamma = decay_step_gamma
        self.decay_step_vf = decay_step_vf    
        self.decay_factor_gamma = decay_factor_gamma
        self.decay_factor_vf = decay_factor_vf

    def update(self, **args):
        raise NotImplementedError


class DifferentialExpTDLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, mu=1, tau=1):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.mu = mu
        self.gamma = 0.5
        self.tau = 1

        self.z = np.exp(-env.R)
        self.z[np.where(self.z > 0)] = 1
        self.N = 50000
        self.rewards = deque(maxlen=50000)


    def update_z(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]

        deltaZ = (w * (np.exp(-r) * self.z[ns] / self.gamma) - self.z[s])

        self.z[s] += self.lr_vf * deltaZ

        if self.samples % self.decay_step_vf == 0:
            self.lr_vf *= self.decay_factor_vf
        
    def update_gamma(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]

        deltaG = (w * (np.exp(-r) * self.z[ns] / self.z[s]) - self.gamma)
        self.gamma += self.mu * self.lr_g * deltaG

        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma


    def get_exp_values(self, list_of_states):

        return [self.z[ns] for ns in list_of_states]

    def get_gain(self):

        return np.log(self.gamma)


class DifferentialLogTDLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, mu=1, tau=1):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.mu = mu
        self.tau = 1

        self.rho = 0
        self.vf = -env.R
        self.vf[np.where(np.exp(self.vf) > 0)] = 0


    def update_z(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]


        delta = (-r - self.rho - w + self.vf[ns] - self.vf[s])
        self.vf[s] += self.lr_vf * delta

        if self.samples % self.decay_step_vf== 0:
            self.lr_vf *= self.decay_factor_vf

    def update_gamma(self, **args):
        
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]

        delta = (-r - self.rho - w + self.vf[ns] - self.vf[s])
        self.rho += self.lr_g * delta

        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma

    def get_exp_values(self, list_of_states):

        return [np.exp(self.vf[ns]) for ns in list_of_states]

    def get_gain(self):
        return self.rho
    
    @property
    def gamma(self):
        return np.exp(self.rho)
    
    @property
    def z(self):

        return np.exp(self.vf)


if __name__ == "__main__":

    algorithm = AbstractAlgorithm()
    algorithm.update()

