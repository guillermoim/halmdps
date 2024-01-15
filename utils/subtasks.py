import numpy as np
from mdps_base.nroom_mdp import create_room_partition
from mdps_base.taxi_mdp import create_taxi_partition


def __init__():
    pass


def learn_subtasks(problem, offset, grid_dim, iters):
    if 'taxi' in problem:
        return learn_subtasks_taxi(offset, grid_dim, iters)
    elif 'nroom' in problem:
        return learn_subtasks_nroom(offset, grid_dim, iters)


def learn_subtasks_taxi(offset, grid_dim, iters=250):

    states, terminal_states, P, R = create_taxi_partition(grid_dim)

    tidxs = list(map(states.index, terminal_states))
    Z = np.ones((len(terminal_states), len(states)))

    mask = np.ones(R.shape, dtype=bool)

    mask[:, tidxs] = False
    R[mask] = - R[mask] - offset
    R[~mask] = - R[~mask]
    Z[:, tidxs] = np.exp(R[:, tidxs])

    G = np.apply_along_axis(np.diag, -1, np.exp(R))

    for i in range(len(terminal_states)):
        Z[i, :] = np.linalg.matrix_power(G[i, :] @ P, iters)  @ Z[i, :]

    return Z, states


def learn_subtasks_nroom(offset=0, grid_dim=3, iters=250):
    """
    TODO: This should not be here.
    """

    assert grid_dim % 2 > 0, "The size of the room shoudl be an odd number."
    goal = (1, 1) if grid_dim == 3 else (2, 3)

    S, TS, P, R = create_room_partition(grid_dim, goal)
    tidxs = list(map(S.index, TS))

    Z = np.ones((len(TS), len(S)))

    mask = np.ones(R.shape, dtype=bool)

    mask[:, tidxs] = False
    R[mask] = - R[mask] - offset
    R[~mask] = - R[~mask]
    Z[:, tidxs] = np.exp(R[:, tidxs])

    G = np.apply_along_axis(np.diag, -1, np.exp(R))

    for i in range(len(TS)):
        Z[i, :] = np.linalg.matrix_power(G[i, :] @ P, iters)  @ Z[i, :]

    return Z, S
