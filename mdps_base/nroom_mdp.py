import numpy as np
import networkx as nx
from itertools import product


def __init__():
    pass


def create_flat_mdp(NUM_ROOMS: tuple,
                    ROOM_SIZE: int,
                    GOAL_POS_INSIDE_ROOM: tuple,
                    GOAL_ROOMS: list,
                    INTERIOR_REWARD: int = 1,
                    GOAL_REWARD: int = 0):

    assert ROOM_SIZE % 2 > 0, "The room size should be an odd number"

    col_rooms, row_rooms = NUM_ROOMS

    X = col_rooms * ROOM_SIZE
    Y = row_rooms * ROOM_SIZE

    graph = nx.grid_graph(dim=[X, Y])

    renaming = {n: (0, *n) for n in graph.nodes()}
    graph = nx.relabel_nodes(graph, renaming)

    # The crossing from one room to another happens at the gridcell in the midpoint
    crossing_point = ROOM_SIZE // 2
    cols = [x * (ROOM_SIZE) - 1 for x in range(1, X)]
    rows = [y * (ROOM_SIZE) - 1 for y in range(1, Y)]

    # Thus, remove any edge that goes "through" the wall.
    for (s, u, v) in graph.nodes():
        if v in cols and (u % ROOM_SIZE != crossing_point) and v != X - 1:
            graph.remove_edge((s, u, v), (s, u, v + 1))
        if u in rows and (v % ROOM_SIZE != crossing_point) and u != Y - 1:
            graph.remove_edge((s, u, v), (s, u + 1, v))

    # Build nre graph.
    graph = nx.DiGraph(graph)

    # Place the top & bottom walls
    for i in range(crossing_point, col_rooms * ROOM_SIZE, ROOM_SIZE):
        graph.add_edge((0, 0, i), (0, -1, i))
        graph.add_edge((0, ROOM_SIZE * row_rooms - 1, i),
                       (0, ROOM_SIZE * row_rooms, i))

    # Place the left and right walls
    for j in range(crossing_point, row_rooms * ROOM_SIZE, ROOM_SIZE):
        graph.add_edge((0, j, 0), (0, j, -1))
        graph.add_edge((0, j, ROOM_SIZE * col_rooms - 1),
                       (0, j, ROOM_SIZE * col_rooms))

    # Add (GOAL) edges.
    for (i, j) in product(range(col_rooms), range(row_rooms)):
        goal_i, goal_j = (ROOM_SIZE * j) + \
            GOAL_POS_INSIDE_ROOM[0], (ROOM_SIZE * i) + GOAL_POS_INSIDE_ROOM[1]
        graph.add_edge((0, goal_i, goal_j), (1, goal_i, goal_j))

    # Add self-loops to the nodes
    for node in graph.nodes():
        graph.add_edge(node, node)

    A = np.asarray(nx.adjacency_matrix(graph).todense())

    P = A * (1 / A.sum(axis=1)[:, None])

    states = list(graph.nodes)
    N = np.product(NUM_ROOMS) * ROOM_SIZE ** 2

    goals_indices = [list(graph.nodes).index(
        (1, ROOM_SIZE * j + GOAL_POS_INSIDE_ROOM[0], ROOM_SIZE * i + GOAL_POS_INSIDE_ROOM[1])) for (i, j) in GOAL_ROOMS]

    interior_states = list(graph.nodes())[:N]
    terminal_states = list(graph.nodes())[N:]
    goal_states = [states[i] for i in goals_indices]

    P[goals_indices, :] = 0
    P[goals_indices, 0] = 1

    # Build reward function
    R = np.full(len(states), INTERIOR_REWARD, dtype=np.float16)
    R[N:] = np.inf

    for i in goals_indices:
        R[i] = GOAL_REWARD

    return interior_states, terminal_states, goal_states, P, R


def create_room_partition(ROOM_SIZE: int,
                          GOAL_POS_INSIDE_ROOM: tuple):

    graph = nx.DiGraph(nx.grid_graph(dim=[ROOM_SIZE, ROOM_SIZE]))

    renaming = {n: (0, *n) for n in graph.nodes()}
    graph = nx.relabel_nodes(graph, renaming)

    # States that are one step away from the terminals.
    terminal_neighbors = [(0, 0, ROOM_SIZE // 2),
                          (0, ROOM_SIZE // 2, 0),
                          (0, ROOM_SIZE // 2, ROOM_SIZE - 1),
                          (0, ROOM_SIZE - 1, ROOM_SIZE // 2),
                          (0, *GOAL_POS_INSIDE_ROOM)]

    terminal_states = [(0, -1, ROOM_SIZE // 2),
                       (0, ROOM_SIZE // 2, -1),
                       (0, ROOM_SIZE // 2, ROOM_SIZE),
                       (0, ROOM_SIZE, ROOM_SIZE // 2),
                       (1, *GOAL_POS_INSIDE_ROOM)]

    # Connect these states.
    for n, t in zip(terminal_neighbors, terminal_states):
        graph.add_edge(n, t)

    for i in graph.nodes():
        graph.add_edge(i, i)

    states = list(graph.nodes())

    A = nx.adjacency_matrix(graph).toarray()
    P = A * (1 / A.sum(axis=1)[:, None])

    R = np.zeros((len(terminal_states), len(states)))
    R[:, :ROOM_SIZE**2] = 1

    for i, t in enumerate(terminal_states):
        R[i, ROOM_SIZE**2:] = np.inf
        R[i, states.index(t)] = 0

    return states, terminal_states, P, R


def get_exit_states(NUM_ROOMS: tuple,
                    ROOM_SIZE: int,
                    GOAL_POS_INSIDE_ROOM: tuple,
                    TS: list) -> dict:

    partitions_set = {}

    d1, d2 = NUM_ROOMS

    for i in range(d2):
        for j in range(d1):
            x, y = i * ROOM_SIZE, j * ROOM_SIZE
            goal_x, goal_y = x + \
                GOAL_POS_INSIDE_ROOM[0], y + GOAL_POS_INSIDE_ROOM[1]
            exit_states = [(0, x - 1, y + ROOM_SIZE // 2), (0, x + ROOM_SIZE // 2, y - 1), (0, x + ROOM_SIZE //
                                                                                            2, y + ROOM_SIZE), (0, x + ROOM_SIZE, y + ROOM_SIZE // 2), (1, goal_x, goal_y)]
            partitions_set[i, j] = {'exit_states': exit_states}

    for key in partitions_set:
        partitions_set[key]['exit_states_inside'] = []

    exit_set = [e for key in partitions_set for e in partitions_set[key]
                ['exit_states'] if e not in TS]

    for e in exit_set:
        partitions_set[get_room(e, ROOM_SIZE)]['exit_states_inside'].append(e)

    return partitions_set


def get_room(state, ROOM_SIZE):

    _, x, y = state

    room = (max(x // ROOM_SIZE, 0), max(y // ROOM_SIZE, 0))

    return room


def project_state(state, ROOM_SIZE):
    s0, s1, s2 = state
    # X, Y = get_room(state, ROOM_SIZE)
    # X, Y = X*ROOM_SIZE, Y*ROOM_SIZE
    return s0, s1 % ROOM_SIZE, s2 % ROOM_SIZE


def unproject_state(state, room, ROOM_SIZE):
    s0, s1, s2 = state
    X, Y = room[0] * ROOM_SIZE, room[1] * ROOM_SIZE

    return s0, X + s1, Y + s2
