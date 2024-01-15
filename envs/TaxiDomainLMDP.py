import numpy as np
import gym

import sys
sys.path.append("..")

from mdps_base.taxi_mdp import *


class TaxiEnvLMDP(gym.Env):

    def __init__(self, DIM=5, PROBLEM_ID=""):

        assert 0 < DIM < 20, "Dimmension must be a integer between 0 and 20"

        self.DIM = DIM

        self.IS, self.TS, self.G, self.T, self.R = create_flat_mdp(self.DIM)

        self.states = self.IS + self.TS

        self.abs_S, self.abs_TS, self.abs_T, self.abs_R = create_taxi_partition(
            DIM)
        
        self.abs_states = self.abs_S + self.abs_TS

        self.current_state = None
        self.current_abstract_state = None
        self.last_partition = None
        self.current_partition = None

        self.partitions = get_exit_states(self.DIM, self.TS)
        self.problem_id = PROBLEM_ID

    def reset(self):
        restart_state = ((0, 0), (0, self.DIM - 1), (self.DIM - 1, 0))
        sampled_state_idx = self.states.index(restart_state)

        self.current_state = self.states[sampled_state_idx]
        self.current_abstract_state = (0, *self.current_state[0])
        self.last_partition = self.current_state[1:]
        self.current_partition = self.current_state[1:]

        return self.get_observation()

    def step(self, next_state):
        self.last_partition = self.current_state[1:]
        reward = self.R[self.states.index(self.current_state)]
        # sidx = self.states.index(self.current_state)
        # nsidx =  # np.where(self.T[action, sidx, :])[0].item()
        # FIXME: This does not use explicit representation of actions
        # (due to the non-symmetry of actions at final state)
        self.current_state = next_state
        self.current_partition = next_state[1:
                                            ] if next_state not in self.TS else self.last_partition
        self.current_abstract_state = (
            0 if next_state not in self.TS else 1, *next_state[0])
        done = self.current_state in self.TS
        info = {'last_partition': self.last_partition,
                'new_partition': self.current_partition}

        return self.get_observation(), reward, done, info

    def get_observation(self):

        return [self.current_state, self.current_partition, (0, *self.current_state[0])]

    def available_actions(self):

        raise NotImplementedError

    def get_next_states(self):
        state = self.current_state
        ns_idxs = np.where(self.T[self.states.index(state), :] > 0)[
            0].tolist()

        return list(map(lambda x: self.states[x], ns_idxs))

    def get_state_idx(self):

        return self.states.index(self.current_state)

    def learn_subtask(self, Z, abs_state, next_abs_state, rho):

        abs_state_idx = None
        next_abs_state_idx = None

        return Z

    def get_partition(self, state):

        return state[1:]

    def project_state_in_partition(self, state, partition):
        k = 0 if state[1:] == partition else 1
        return (k, *state[0])


    def get_LMDP(self):

        return self.IS, self.TS, self.G, self.T, self.R
    
    @property
    def exit_states(self):

        H = self.partitions

        EXIT_STATES = list(set(E for p in H for E in H[p]["exit_states"]))

        return EXIT_STATES