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


class DifferentialZLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, eta=1, mu=1, fixed_gamma=False):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.gamma = 1
        self.tau = 1
        self.eta = eta
        self.mu = mu

        self.z = np.exp(-self.eta * env.R)
        self.z[np.where(self.z > 0)] = 1
        self.fixed_gamma = fixed_gamma
        if self.fixed_gamma:
            self.gamma = fixed_gamma

    def update(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]
        update_vf = args["update_vf"]

        deltaZ = (w * (np.exp(-self.eta* r) * self.z[ns] / self.gamma) - self.z[s])
        deltaG = (w * (np.exp(-self.eta * r) * self.z[ns] / self.z[s]) - self.gamma)

        self.gamma += self.mu * self.lr_g * deltaG
        
        if update_vf:
            self.z[s] += self.lr_vf * deltaZ

        if self.samples % self.decay_step_vf == 0:
            self.lr_vf *= self.decay_factor_vf
        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma
        
    def update_gamma(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]

        if not self.fixed_gamma:
            deltaG = (w * (np.exp(-self.eta * r) * self.z[ns] / self.z[s]) - self.gamma)
            self.gamma += self.mu * self.lr_g * deltaG

        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma


    def get_exp_values(self, list_of_states):

        return [self.z[ns] for ns in list_of_states]

    def get_gain(self):

        return np.log(self.gamma) / self.eta


class DifferentialLogTDLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, mu=1, tau=1, eta=1):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.mu = mu
        self.tau = 1

        self.eta = eta

        self.rho = 0
        self.vf = -env.R
        self.vf[np.where(np.exp(self.vf) > 0)] = 0

    def update(self, **args):
        s = self.env.states.index(args["state"])
        r = args["reward"]
        update_vf = args["update_vf"]

        delta = (-r + ( -self.rho + np.log(np.dot(self.env.T[s], self.z)) - self.vf[s]) / self.eta)
        self.rho += self.lr_g * delta

        if update_vf:
            self.vf[s] += self.lr_vf * delta

        if self.samples % self.decay_step_vf== 0:
            self.lr_vf *= self.decay_factor_vf

        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma

    def get_exp_values(self, list_of_states):

        return [np.exp(self.vf[ns]) for ns in list_of_states]

    def get_gain(self):
        return self.rho
    
    @property
    def gamma(self):
        return np.exp(self.eta * self.rho)
    
    @property
    def z(self):
        return np.exp(self.eta * self.vf)


if __name__ == "__main__":

    algorithm = AbstractAlgorithm()
    algorithm.update()


class DifferentialExpTDLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, mu=1, tau=1, eta=1):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.mu = mu
        self.tau = 1

        self.eta = eta

        self.gamma = 1
        self.z = np.exp(-env.R)
        self.z[np.where(self.z > 0)] = 1

    def update(self, **args):
        s = self.env.states.index(args["state"])
        r = args["reward"]
        update_vf = args["update_vf"]

        delta = (-r + ( -self.rho + np.log(np.dot(self.env.T[s], self.z)) - self.vf[s]) / self.eta)
        self.gamma *= np.exp(self.eta * delta) ** self.lr_g

        if update_vf:
            self.z[s] *= np.exp(self.eta * delta) ** self.lr_vf

        if self.samples % self.decay_step_vf== 0:
            self.lr_vf *= self.decay_factor_vf

        if self.samples % self.decay_step_gamma == 0:
            self.lr_g *= self.decay_factor_gamma

    def get_exp_values(self, list_of_states):

        return [np.exp(self.vf[ns]) for ns in list_of_states]

    def get_gain(self):
        return self.rho
    
    @property
    def rho(self):
        return (1 / self.eta) * np.log(self.gamma)
    
    @property
    def vf(self):
        return (1/self.eta) * np.log(self.z)


if __name__ == "__main__":

    algorithm = AbstractAlgorithm()
    algorithm.update()
