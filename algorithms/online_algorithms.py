from collections import defaultdict
import numpy as np


class AbstractAlgorithm:

    def __init__(self, 
                 env,
                 decay_step_gamma: int,
                 decay_step_z: int,
                 decay_factor_gamma: float,
                 decay_factor_z: float,
                 init_lr_z: int = 1,
                 init_lr_g: int = 1):

        self.env = env
        self.samples = 0
        self.lr_z = init_lr_z
        self.lr_g = init_lr_g
        self.decay_step_gamma = decay_step_gamma
        self.decay_step_z = decay_step_z      
        self.decay_factor_gamma = decay_factor_gamma
        self.decay_factor_z = decay_factor_z

    def update(self, **args):
        raise NotImplementedError


class DifferentialExpTDLearning(AbstractAlgorithm):

    def __init__(self, env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z, mu=1, tau=1):
        super().__init__(env, decay_step_gamma, decay_step_z, decay_factor_gamma, decay_factor_z)
        self.mu = mu
        self.gamma = 1
        self.tau = 1

        self.z = np.exp(-env.R)
        self.z[np.where(self.z > 0)] = 1


    def update_z(self, **args):
        s = self.env.states.index(args["state"])
        ns = self.env.states.index(args["next_state"])
        r = args["reward"]
        w = args["isw"]

        deltaZ = (w * (np.exp(-r) * self.z[ns] / self.gamma) - self.z[s])

        self.z[s] += self.lr_z * deltaZ

        if self.samples % self.decay_step_z == 0:
            self.lr_z *= self.decay_factor_z
        
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


# class DifferentialTDLearning(AbstractAlgorithm):

#     def __init__(self, decay_step, init_lr_vf=1, init_lr_g=1, mu=1, tau=1):
#         super().__init__(decay_step=decay_step)
#         self.mu = mu
#         self.vf = defaultdict(lambda: 1)
#         self.gamma = 1
#         self.tau = 1

#     def set_vf_terminal_states(self, states, rewards):
#         for i, state in enumerate(states):
#             self.vf[state] = np.exp(rewards[i])

#     def update(self, **args):
#         s = args["state"]
#         nss = args["next_states"]
#         ns = args["next_state"]
#         r = args["reward"]

#         sum_exp_values = np.mean(
#             [self.vf[ns] for ns in nss])

#         delta = -r - np.log(self.gamma) + \
#             np.log(sum_exp_values) - np.log(self.vf[s])

#         self.vf[s] *= np.exp(self.lr_vf * delta)
#         self.gamma *= np.exp(self.mu * self.lr_g * delta)

#         self.samples += 1

#         if self.samples % self.decay_step == 0:
#             self.lr_vf *= self.decay_factor_vf
#             self.lr_g *= self.decay_factor_gamma

#     def get_exp_values(self, list_of_states):

#         return [self.vf[ns] for ns in list_of_states]

#     def get_gain(self):

#         try:
#             return np.log(self.gamma)
#         except:
#             print('Gamma', self.gamma)
#             exit()


if __name__ == "__main__":

    algorithm = AbstractAlgorithm()
    algorithm.update()

