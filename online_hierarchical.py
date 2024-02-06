import numpy as np
import sys 
sys.path.append('../envs')
import envs
import gym

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.learning_rate_scheduler import LearningRateScheduler
from utils.subtasks import learn_subtasks 

import pickle as pkl

import wandb
from tqdm import tqdm

from algorithms.hierarchical_online_algorithms import HierarchicalExp


def online(env, eta, algo, SAMPLES=1e5):

    # Retrieve optimal solutions
    with open(f"results/ground_truth/{env.unwrapped.spec.id}_{eta:.4f}.pkl", "rb") as fp:

       Z_OPT, GAMMA_OPT, ZS_OPT, _ =  pkl.load(fp)

    # Prepare environment and initialize parameters

    REF_STATE = env.G[0]


    # Initialize values value function of the exit states

    state, _, abs_state = env.reset()

    for i in tqdm(range(int(SAMPLES)), disable=False):

        # Algorithm 2 - Line 4: Sample state according to current estimates and get importance sampling weight
        next_state, isw = algo.sample_next_state(state)

        # Algorithm 2 - Line 5: Update subtasks' value functions        
        algo.update_subtasks(abs_state = abs_state, update=state not in env.G)
            
        # Make transition in the LMDP and get new observation
        (next_state, _, abs_state), reward, _, _ = env.step(next_state)

        # Algorithm 2 - Line 6: Update estimated gamma and value function Z if corresponds
        algo.update(state = state, 
                    next_state = next_state,
                    isw = isw, 
                    reward = reward, 
                    update_vf = state != REF_STATE)


        state = next_state

        algo.update_lrs()


        # Log into wandb
        if i % 200 == 0:
            
            ERROR_GAMMA = np.abs(GAMMA_OPT - algo.gamma)
            ERROR_Zs = np.abs(algo.subtasks - ZS_OPT).mean()
            ERROR_Z  = np.mean(np.abs(Z_OPT[algo.exit_states_idxs] - algo.exit_estimates))

            # print(ERROR_GAMMA, ERROR_Zs, ERROR_Z)

            log_dict = {"train/gamma": algo.gamma,
                        "train/MAE_subtasks": ERROR_Zs,
                        "train/MAE_exit_states": ERROR_Z,
                        "train/Error_Gamma": ERROR_GAMMA,
                        "train/gt_gamma": GAMMA_OPT,
                        "learning_rates/z": algo.lrs.lr1,
                        "learning_rates/subtasks": algo.lrs.lr2,
                        "learning_rates/gamma": algo.lrs.lr3,
                        "step": i}
            
            wandb.log(log_dict)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:


    t = "log" if 'Log' in cfg.algorithm["_target_"] else "exp"
    
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project,
        group=f"{cfg.env_name}-{t}-hierarchical", 
        tags=["h-online"],
        # mode="disabled" 
    )

    env_name = cfg.env_name
    env = gym.make(env_name)

    LRS = LearningRateScheduler(**cfg.lrs)
    
    algo = hydra.utils.call(config=cfg.algorithm, env=env, lrs=LRS)

    online(env, cfg.algorithm["eta"], algo, int(cfg.n_samples))

    wandb.finish()

if __name__ == "__main__":

    main()