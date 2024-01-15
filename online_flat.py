
from mdps_base.problem_loader import *
from algorithms.online_algorithms import *
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


def online(env, SAMPLES, algo, seed):

    with open(f"results/ground_truth/{env.unwrapped.spec.id}.pkl", "rb") as fp:

       Z_OPT, GAMMA_OPT, ZS_OPT =  pkl.load(fp)

    REF_STATE = env.G[0]

    np.random.seed = seed

    # Set the value function of the terminal states equal to the reward

    (state, _, _) = env.reset()


    for i in tqdm(range(SAMPLES)):

        # Get the current state and next states' values.
        next_states_idxs = np.where(env.T[env.states.index(state), :] > 0)[0].tolist()
        # next_states = list(map(lambda x: env.states[x], next_states))
        next_states_values = algo.get_exp_values(next_states_idxs)

        pi = next_states_values / np.sum(next_states_values)

        action = np.random.choice(len(next_states_idxs), p=pi)
        next_state = env.states[next_states_idxs[action]]

        (next_state, _, _), reward, _, _ = env.step(next_state)
        algo.samples+=1

        isw = (1 / len(next_states_idxs)) / pi[action]

        algo.update_gamma(state=state, next_state=next_state, reward=reward, isw=isw)

        if state != REF_STATE:

            algo.update_z(
                state=state, next_state=next_state, reward=reward, isw=isw)


        state = next_state

        if i % 100 == 0:

            log_dict = {"train/gamma": algo.gamma,
                        "train/MAE_z": np.abs(Z_OPT - algo.z).mean(),
                        "train/Error_Gamma": np.abs(GAMMA_OPT - algo.gamma),
                        "train/gt_gamma": GAMMA_OPT,
                        "learning_rates/z": algo.lr_z,
                        "learning_rates/gamma":  algo.lr_g,
                        "step": i}
            
            wandb.log(log_dict)

    return algo.z


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project,
        group=f"{cfg.env_name}-flat", 
        tags=["f-online"],
        # mode="disabled"
        )


    env_name = cfg.env_name
    env = gym.make(env_name)


    algo = DifferentialExpTDLearning(env, **cfg.lrs)


    online(env, int(1e6), algo, 42)

    
    wandb.finish()


if __name__ == "__main__":


    main()

    wandb.finish()
