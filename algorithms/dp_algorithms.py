import numpy as np

import sys
sys.path.append('..')

from tqdm import tqdm


def __init__():
    pass


def exp_rvi(S: list, P: np.ndarray, R: np.ndarray, eta=1, iters: int = 1000, with_gains = False):
    """
        Run Relative Value Iteration in the exponentiated space.

        Returns the value function 'normalized' with the state at index 0.

    """
    # TODO: Add temperature parameter
    z = np.ones(len(S))
    G = np.diagflat(np.exp(- eta* R))
    gammas = []

    for _ in tqdm(range(iters)):

        z = G @ P @ z
        gamma = z[0]
        z = z / z[0]
        gammas.append(gamma)

    if with_gains:
        return z, gamma, np.asarray(gammas)
    else:
        return z, gamma

def rvi(S: list, P: np.ndarray, R: np.ndarray, iters: int = 10000):

    # TODO: Add temperature parameter
    """
        This applies the relative Bellman optimality equation as in
        Todorov's paper in the non-exponential space.

        Corresponds to Eq. 3 in 'Policy gradients in linearly-solvable MDPs'.

        Returns the value function 'normalized' with the state at index 0.

        URL: https://homes.cs.washington.edu/~todorov/papers/TodorovNIPS10.pdf
    """

    V = np.zeros(len(S))

    for _ in tqdm(range(iters)):

        ref_value = V[0]

        # First iteration of the loop -> use all states
        # This implies: for terminal states it assigns the rewards as the value function.
        # Otherwise, apply the update rule.
        idxs = np.where(~(P @ np.exp(V) == 0))[0]

        V[idxs] = - R[idxs] + \
            np.log((P[idxs, :] @ np.exp(V)))

        rho = V[0] - ref_value
        V = V - V[0]

    return V, rho
