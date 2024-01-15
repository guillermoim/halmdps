import numpy as np
import networkx as nx
from itertools import product

# actions constant variables
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
NO_OP = 4
ACTION = 5

N_ACTIONS = 6


def __init__():
    pass


def create_flat_mdp(DIM: int = 5,
                    INTERIOR_REWARD: int = 1,
                    GOAL_REWARD: int = 0):
    '''
    This method creates the flat MDP for our taxi domain.

    States are represented by a tuple (grid_location, passenger_location, final_destination), thus the information
    for the goal is included in the state representation.

    The dynamics can be thought of different gridworlds in which first the (sub)goal is determined by the passenger location
    and when he/she is picked-up, the final destination conditions the next gridworld that is accessed right after in which
    such destination is the final goal.
    '''

    nav_locs = [(i, j) for i in range(DIM) for j in range(DIM)]

    corners = [(0, 0), (DIM - 1, 0), (0, DIM - 1), (DIM - 1, DIM - 1)]
    passenger_locs = corners + ['TAXI']

    # create possible transitions
    loc_and_neighbors = {}

    for loc in nav_locs:
        neighbors = []
        # UP and LEFT
        if loc[0] - 1 > -1:
            neighbors.append((loc[0] - 1, loc[1]))
        if loc[1] - 1 > -1:
            neighbors.append((loc[0], loc[1] - 1))
        # DOWN and RIGHT
        if loc[0] + 1 < DIM:
            neighbors.append((loc[0] + 1, loc[1]))
        if loc[1] + 1 < DIM:
            neighbors.append((loc[0], loc[1] + 1))

        loc_and_neighbors[loc] = neighbors

    transitions_1 = []
    transitions_2 = []

    # First of all, add the transition between the (exit)states. This is between the pick-up
    # locations and the next state in which the agent is in the TAXI.
    for c0 in corners:
        for c1 in corners:
            if c0 == c1:
                continue
            # exit state transition
            transition = (c0, c0, c1), (c0, 'TAXI', c1)
            transition_reversed = (c0, 'TAXI', c1), (c0, c0, c1)
            transitions_1.append(transition)
            transitions_1.append(transition_reversed)

    # Then, add the transitions regarding the navigation within same gridworlds.
    for c0 in passenger_locs:
        for c1 in corners:
            if c0 == c1:
                continue
            for xy in nav_locs:
                for neighbor in loc_and_neighbors[xy]:
                    transition = (xy, c0, c1), (neighbor, c0, c1)
                    transition_reversed = (neighbor, c0, c1), (xy, c0, c1)
                    transitions_2.append(transition)
                    transitions_2.append(transition_reversed)

    terminal_edges = []

    # Then, for each terminal goal I add a directed edge in which the passenger is at final destination.
    for corner in corners:
        transition = (corner, 'TAXI', corner), (corner, 'D', corner)
        terminal_edges.append(transition)

    # Also, I need to add some terminals in the 'first' gridworlds to allow exploration, these terminals happen
    # (taxi_loc, pass, dst) whenever taxi_loc = corner and taxi_loc != pass.
    # only if so specified by argument <terminal_non_goals>
    for taxi in corners:
        for passenger in corners:
            for dst in corners:
                if taxi == passenger:
                    continue
                if passenger == dst:
                    continue
                transition = (taxi, passenger, dst), (taxi,
                                                      'Forbidden', None)
                terminal_edges.append(transition)

    graph = nx.DiGraph()
    graph.add_edges_from(transitions_1)
    graph.add_edges_from(transitions_2)
    graph.add_edges_from(terminal_edges)

    for node in graph.nodes():
        graph.add_edge(node, node)

    states = list(graph.nodes())

    # sample_states = [s for s in states if s[1]
    #                  not in ('D', 'TAXI', 'Forbidden')]

    goal_states = [s for s in states if s[1] == 'D']
    non_goal_states = [s for s in states if s[1] == 'Forbidden']

    A = np.asarray(nx.linalg.adjacency_matrix(graph).toarray())
    
    P = A * (1 / A.sum(axis=1)[:, None])

    R = np.ndarray(P.shape[0], dtype=np.float64)

    terminal_states = goal_states + non_goal_states
    interior_states = [s for s in states if s not in terminal_states]

    R[[states.index(s) for s in interior_states]] = INTERIOR_REWARD
    R[[states.index(s) for s in goal_states]] = GOAL_REWARD
    R[[states.index(s) for s in non_goal_states]] = np.inf

    P = np.asarray(P)

    for g in goal_states:

        taxi_loc = g[0]
        reset_states = [t for t in states if t[0] == taxi_loc and t[1]
                        not in ('TAXI', 'Forbidden', 'D')]

        idxs = list(map(states.index, reset_states))

        P[states.index(g), :] = 0
        P[states.index(g), idxs] = 1 / len(idxs)

    G = list(map(lambda x: states[x], np.where(R == 0)[0].tolist()))
    # IS, TS, G, T, R
    return interior_states, terminal_states, G, P, R


def create_taxi_partition(DIM):

    # Returns a taxi room with 4 terminal states.
    graph = nx.grid_graph((DIM, DIM)).to_directed()
    graph = nx.DiGraph(graph)
    # Change node names from (x,y) to (0,x,y) so we can have terminals @ (1,0,0) ... (1,r_DIM-1, r_DIM-1)
    mapping = {node: (0, *node) for node in graph.nodes}

    graph = nx.relabel_nodes(graph, mapping)

    graph.add_edge((0, 0, 0), (1, 0, 0))
    graph.add_edge((0, DIM - 1, 0), (1, DIM - 1, 0))
    graph.add_edge((0, 0, DIM - 1), (1, 0, DIM - 1))
    graph.add_edge((0, DIM - 1, DIM - 1), (1, DIM - 1, DIM - 1))

    for node in graph.nodes:
        graph.add_edge(node, node)

    states = list(graph.nodes)

    A = nx.adjacency_matrix(graph).toarray()

    P = A * (1 / A.sum(axis=1)[:, None])

    R = np.full((4, DIM**2 + 4), 1, dtype=np.float64)
    R[:, -4:] = np.inf

    for i, t in enumerate(states[-4:]):
        idx = states.index(t)
        R[i, idx] = 0

    P = np.asarray(P)

    terminal_idxs = [states.index(s) for s in states if s[0] == 1]
    terminals = list(map(lambda x: states[x], terminal_idxs))

    return states, terminals, P, R


def get_exit_states(DIM, T):

    partitions_set = {}

    corners = [(0, 0), (DIM - 1, 0), (0, DIM - 1), (DIM - 1, DIM - 1)]
    passenger_locs = corners + ['TAXI']

    for c1 in passenger_locs:
        for c2 in corners:
            if c1 != c2:
                partition = (c1, c2)
                if c1 == 'TAXI':
                    aux = [(c, c, c2)
                           for c in corners if c != c2] + [(c2, 'D', c2)]
                    exit_states = [None, None, None, None]
                    for e in aux:
                        if e[1] != 'D':
                            exit_states[corners.index(e[1])] = e
                        else:
                            exit_states[corners.index(e[2])] = e
                else:
                    aux = [(c, 'Forbidden', None)
                           for c in corners if c != c1] + [(c1, 'TAXI', c2)]
                    exit_states = [None, None, None, None]
                    for e in aux:
                        exit_states[corners.index(e[0])] = e

                partitions_set[partition] = {'exit_states': exit_states}

    exit_set = [e for key in partitions_set for e in partitions_set[key]
                ['exit_states'] if e not in T]

    for key in partitions_set:
        partitions_set[key]['exit_states_inside'] = []

    exit_set = [e for key in partitions_set for e in partitions_set[key]
                ['exit_states'] if e not in T]

    for e in exit_set:
        partition = e[1], e[2]
        partitions_set[partition]['exit_states_inside'].append(e)

    return partitions_set
