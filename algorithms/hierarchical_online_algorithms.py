import numpy as np



class HierarchicalExp:


    def __init__(self, env, lrs):

        self.env = env

        self.zs = None

        self.exit_estimates

        self.LRS = lrs


    def update(self, **args):
        
        s = self.env.states.index(args["state"])
        abs_state = args["abs_state"]
        next_state = args["next_state"]
        partition = args["partition"]

        r = args["reward"]
        w = args["isw"]
        update_vf = args["update_vf"]

        abs_ns = self.env.project_state_in_partition(next_state, partition)

        abs_s_idx = self.env.abs_states.index(abs_state)


    def _update_subtasks(self):
        pass
    

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
        
    def _prepare_subtasks(self):
        pass


    def get_policy(self, state):

        pass