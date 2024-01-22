import numpy as np
import pickle as pkl
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str)

    args = parser.parse_args()
    env_name = args.env_name



    with open(f"results/ground_truth/{env_name}.pkl", "rb") as fp:
        
        Z, gamma, zs = pkl.load(fp)


        print(gamma)
        print(np.where(Z==1))
