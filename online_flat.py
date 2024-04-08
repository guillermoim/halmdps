
from mdps_base.problem_loader import *
from algorithms.flat_online_algorithms import *
import pickle as pkl
import numpy as np
import pickle as pkl
# from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from tqdm import tqdm

import gym 
import envs

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
import pandas as pd


def online(env, SAMPLES, algo, eta, seed):

    with open(f"results/ground_truth/{env.unwrapped.spec.id}_{eta:.4f}.pkl", "rb") as fp:

       Z_OPT, GAMMA_OPT, _, eta =  pkl.load(fp)

    REF_STATE = env.G[0]

    np.random.seed = seed

    # Set the value function of the terminal states equal to the reward

    (state, _, _) = env.reset()


    for i in tqdm(range(SAMPLES)):

        # Get the current state and next states' values.
        next_states_idxs = np.where(env.T[env.states.index(state), :] > 0)[0].tolist()
        # next_states = list(map(lambda x: env.states[x], next_states))
        next_states_values = algo.z[next_states_idxs]

        pi = next_states_values / np.sum(next_states_values)

        action = np.random.choice(len(next_states_idxs), p=pi)
        next_state = env.states[next_states_idxs[action]]

        (next_state, _, _), reward, _, _ = env.step(next_state)
        algo.samples+=1

        isw = (1 / len(next_states_idxs)) / pi[action]

        algo.update(state=state, next_state=next_state, reward=reward, isw=isw, update_vf=not state in env.G)

        state = next_state

        if i % 1000 == 0:

            log_dict = {"train/gamma": algo.gamma,
                        "train/MAE_z": np.abs(Z_OPT - algo.z).mean(),
                        "train/Error_Gamma": np.abs(GAMMA_OPT - algo.gamma),
                        "train/gt_gamma": GAMMA_OPT,
                        "learning_rates/z": algo.lr_vf,
                        "learning_rates/gamma":  algo.lr_g,
                        "step": i,}
            
            wandb.log(log_dict)
    
    df = pd.DataFrame({"z":algo.z, "z_opt": Z_OPT})
    table = wandb.Table(dataframe=df)

    artifact = wandb.Artifact("value_functions", type="model")
    artifact.add(table, "value_functions_table")

    wandb.log_artifact(artifact)
    
    return algo.z


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:

    if 'Log' in cfg.algorithm["_target_"]:
        t = "log"  
    elif "ExpTD" in cfg.algorithm["_target_"]: 
        t = "exp-td"
    else:
        t = "exp"

    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project,
        group=f"{cfg.env_name}-{t}-flat", 
        tags=["f-online"],
        # mode="disabled"
        )


    env_name = cfg.env_name
    env = gym.make(env_name)
    eta = cfg.algorithm["eta"]

    algo = hydra.utils.call(config=cfg.algorithm, env=env)

    # algo = DifferentialExpTDLearning(env, **cfg.lrs)

    online(env, int(cfg.n_samples), algo, eta, 42)

    wandb.finish()


if __name__ == "__main__":


    main()

    wandb.finish()
