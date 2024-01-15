import numpy as np
import gym

import sys
sys.path.append("..")

from mdps_base.nroom_mdp import *


class NRoomEnvLMDP(gym.Env):

    def __init__(self, NUM_ROOMS=(2, 2), DIM=5, GOAL_POS_INSIDE_ROOM=(2, 3), GOAL_ROOMS=[(1, 1)], PROBLEM_ID=""):
        assert type(NUM_ROOMS) == tuple
        assert len(NUM_ROOMS) == 2
        assert type(DIM) == int and DIM >= 3

        self.DIM = DIM

        self.IS, self.TS, self.G, self.T, self.R = create_flat_mdp(
            NUM_ROOMS, DIM, GOAL_POS_INSIDE_ROOM, GOAL_ROOMS)

        self.states = self.IS + self.TS

        self.abs_S, self.abs_TS, self.abs_T, self.abs_R = create_room_partition(
            DIM, GOAL_POS_INSIDE_ROOM)
        
        self.abs_states = self.abs_S + self.abs_TS

        self.problem_id = PROBLEM_ID

        self.current_state = None
        self.current_abstract_state = None
        self.last_partition = None
        self.current_partition = None

        self.partitions = get_exit_states(
            NUM_ROOMS, DIM, GOAL_POS_INSIDE_ROOM, self.TS)

    def reset(self):

        sampled_state_idx = self.states.index((0, 0, 0))

        self.current_state = self.states[sampled_state_idx]
        self.current_abstract_state = project_state(
            self.current_state, self.DIM)
        self.last_partition = get_room(self.current_state, self.DIM)
        self.current_partition = get_room(self.current_state, self.DIM)

        return self.get_observation()

    def step(self, next_state):
        self.last_partition = get_room(self.current_state, self.DIM)
        reward = self.R[self.states.index(self.current_state)]
        # sidx = self.states.index(self.current_state)
        # nsidx =  # np.where(self.T[action, sidx, :])[0].item()
        # FIXME: This does not use explicit representation of actions
        # (due to the non-symmetry of actions at final state)
        self.current_state = next_state
        self.current_partition = get_room(next_state, self.DIM)
        self.current_abstract_state = project_state(
            self.current_state, self.DIM)
        done = self.current_state in self.TS
        info = {'last_partition': self.last_partition,
                'new_partition': get_room(next_state, self.DIM)}

        return self.get_observation(), reward, done, info

    def get_observation(self):

        return [self.current_state, get_room(self.current_state, self.DIM), project_state(self.current_state, self.DIM)]

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

        return get_room(state, self.DIM)

    def project_state_in_partition(self, state, partition):
        room_dim = self.DIM
        (k, x, y) = state
        x, y = x - partition[0] * room_dim, y - partition[1] * room_dim
        return (k, x, y)

    def get_LMDP(self):

        return self.IS, self.TS, self.G, self.T, self.R
    
    @property
    def exit_states(self):

        H = self.partitions

        EXIT_STATES = list(set(E for p in H for E in H[p]["exit_states"]))

        return EXIT_STATES