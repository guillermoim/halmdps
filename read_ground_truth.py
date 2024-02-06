import numpy as np
import pickle as pkl
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--eta", default=1)


    args = parser.parse_args()
    env_name = args.env_name
    eta = args.eta

    with open(f"results/ground_truth/{env_name}_{eta:.4f}.pkl", "rb") as fp:
        
        Z, gamma, zs, _ = pkl.load(fp)


        print(gamma, np.log(gamma))
        print(Z.shape)
        print(zs)
