import gym 
import envs

from algorithms.rvi import solve as solve_flat
from utils.subtasks import learn_subtasks
import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt

MAX_ITERS = 50
EPS = 1e-8

def _setup_problem(env, zs, ABS_STATES, REF_STATE):

    H = env.partitions
    EXIT_STATES = list(set(E for p in H for E in H[p]["exit_states"]))

    ne = len(EXIT_STATES)
    
    Ze = np.ones(ne)

    for i, e in enumerate(EXIT_STATES):
        if e in env.TS:
            Ze[i] = np.exp(-env.R[env.states.index(e)])

    Ge = np.zeros((ne, ne))

    for h in H:
        idxs = list(map(EXIT_STATES.index, H[h]["exit_states"]))
        for e in H[h]['exit_states_inside']:
            abs_idx = ABS_STATES.index(env.project_state_in_partition(e, h))
            Ge[EXIT_STATES.index(e), idxs] = zs[:, abs_idx]

    for i, e in enumerate(EXIT_STATES):
        if e in env.TS:
            Ge[i, i] = 1

    REF_STATE_idx = EXIT_STATES.index(REF_STATE)

    return Ge, Ze, REF_STATE_idx


def get_explicit_value(s, env, zs, ABS_STATES, Ze, fn = lambda x: (0, *x[0])):

    H = env.partitions
    EXIT_STATES = list(set(E for p in H for E in H[p]["exit_states"]))

    P = env.T[env.states.index(s), :]
    next_states = np.where(P > 0)[0].tolist()

    Z = np.zeros(P.shape)

    for i in next_states:
        state = env.states[i]
        p = env.get_partition(state)
        eidxs = [EXIT_STATES.index(e) for e in H[p]["exit_states"]]
        abs_idx = ABS_STATES.index(env.project_state_in_partition(state, p))
        Z[i] = np.dot(Ze[eidxs], zs[:, abs_idx])

    return np.dot(P, Z)

def solve_hierarchical(env, low_level_iters):

    # Take an arnitrary 'goal' state as REF_STATE
    REF_STATE = env.G[0]

    lo, hi = 0, 1

    gammas, lows, highs, means = [], [], [], []

    while np.abs(hi - lo) > EPS:

        gamma = (lo + hi) / 2

        # Algorithm 1: Line 5 - Solve the substasks for the current estimation of gamma
        zs, ABS_STATES = learn_subtasks(env.problem_id, np.log(gamma), env.DIM, low_level_iters)

        # Algorithm 1: Line 6 - Construct matrix
        Ge, Ze, REF_STATE_idx = _setup_problem(env, zs, ABS_STATES, REF_STATE)

        # Algorithm 1: Line 7: Solve system of linear eqs. Ze = Ge @ Ze
        for _ in range(MAX_ITERS):

            Ze = Ge @ Ze
            Ze /= Ze[REF_STATE_idx]

        # Algorithm 1: Lines 8 - 11, compare the explicit value of REF_STATE and compare against estimated value
        comp_value = np.exp(-env.R[env.states.index(REF_STATE)]) * get_explicit_value(REF_STATE, env, zs, ABS_STATES, Ze)

        gammas.append(gamma)
        lows.append(lo)
        highs.append(hi)
        means.append(np.nanmean(Ze))
        
        if comp_value / gamma < 1:
            hi = gamma
        else:
            lo = gamma

    return gammas, lows, highs, means


def high_level_solver():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--low_level_iters", type=int, default=5000)

    args = parser.parse_args()
    env_id = args.env
    low_level_iters = args.low_level_iters

    env = gym.make(env_id)

    z, true_gamma, zs = solve_flat(env)

    gammas, lows, highs, means = solve_hierarchical(env, low_level_iters)

    errors = np.abs(gammas - true_gamma)

    fig, ax = plt.subplots(1,1, figsize=(9, 6))
    ax.set_title(fr'Eigenvectors - {env.problem_id} - MAE={errors[-1]:.6f}')
    ax.plot(lows, linewidth=1.5, color='red', label='low')
    ax.plot(highs, linewidth=1.5, color='blue', label='highs')
    ax.plot(gammas, linewidth=1.5, color='black', label=fr'$\Gamma$s')
    ax.axhline(true_gamma, linewidth=1.5, color='green', linestyle="dashed", label=fr'True $\Gamma$')
    ax.plot(errors, linewidth=1.5, color='pink', linestyle="dotted", label=fr'MAE$')
    ax.legend(fontsize=18, loc=1)

    plt.savefig(f'results/hierarchical/eigenvectors/{env.problem_id}.pdf', bbox_inches='tight', dpi=500)
