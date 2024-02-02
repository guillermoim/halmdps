import sys
sys.path.append('..')

import pickle as pkl
import numpy as np
import argparse

from utils.subtasks import *

from algorithms.dp_algorithms import *
from mdps_base.problem_loader import load_problem, list_problems
from collections import defaultdict
import os

import gym 
import envs


def power_method(P, G, n_iter=1000):
    z = np.ones((G.shape[0], 1))

    P = np.asarray(P)
    G = np.asarray(G)

    for _ in range(n_iter):
        z = np.dot(np.matmul(G, P), z)
        z /= np.linalg.norm(z)

    return z


def solve(env, eta=1, iters=10000, log=False):
 
    IS, TS, _, P, R = env.get_LMDP()

    # Run RVI
    z, gain = exp_rvi(IS + TS, P, R, eta=eta, iters=iters)

    # Get eigenvectors
    w, v = np.linalg.eig(np.diag(np.exp(-eta * R)) @ P)
    gammaeig = np.max(w)
    zeig = v[:, np.argmax(w)]

    gamma = np.exp(gain)
    
    rsidx = env.states.index(env.G[0])

    z /= z[rsidx]
    zeig /= zeig[rsidx]

    if log:
        print('Gain RVI:', gain, "Gamma RVI:", np.exp(gain))
        print('Max eigenvalue:', gammaeig, "log(eig):", np.log(gammaeig))
        print('Error gamma', np.abs(np.exp(gain) - gammaeig))
        print('Error gain', np.abs(gain - (np.log(gammaeig) ) ))
        print('Error in z', np.abs(z - zeig).mean())
        print('Error in v', np.abs(
            np.log(z[:len(IS)]) - np.log(zeig[:len(IS)])).mean())


    zs, _ = learn_subtasks(env.problem_id, np.log(gamma), env.DIM, 10000)

    return z, gamma, zs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--eta", type=float, default=1)

    args = parser.parse_args()

    env_name = args.env_name
    eta = args.eta


    env = gym.make(env_name)

    Z, gamma, zs = solve(env, eta=eta, log=True)

    os.makedirs("../results/ground_truth/", exist_ok=True)

    with open(f"../results/ground_truth/{env_name}_{eta:.4f}.pkl", "wb") as fp:

        pkl.dump((Z, gamma, zs, eta), fp)