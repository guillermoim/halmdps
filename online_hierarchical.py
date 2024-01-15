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

# Set initial value of gamma

def get_composed_value(state, env, subtasks, exit_estimates):
    
    if state in env.TS:
        
        return exit_estimates[env.exit_states.index(state)]
    
    else:
        
        partition = env.get_partition(state)
        abs_state = env.project_state_in_partition(state, partition)
        exit_states_idxs = [env.exit_states.index(e) for e in env.partitions[partition]['exit_states']]
        w = exit_estimates[exit_states_idxs]
        z = np.dot(w, subtasks)

        return z[env.abs_states.index(abs_state)]

def algorithm(env, LRS, SAMPLES=1e6):

    # Retrieve optimal solutions
    with open(f"results/ground_truth/{env.unwrapped.spec.id}.pkl", "rb") as fp:

       Z_OPT, GAMMA_OPT, ZS_OPT =  pkl.load(fp)

    # Prepare environment and initialize parameters

    REF_STATE = env.G[0]

    # Initialize subtasks' value functions
    Zs, _ = learn_subtasks(env.problem_id, 0, env.DIM, 1)
    Zs[np.where(Zs > 0)] = 1 / len(Zs)

    # Initialize values value function of the exit states
    exit_states_idxs = list(map(env.states.index, env.exit_states))
    exit_estimates = np.exp(-env.R[exit_states_idxs])

    f_aux = lambda x: get_composed_value(x, env, Zs, exit_estimates)

    gamma = 1

    # Reset environment
    state, partition, abs_state = env.reset()

    for i in range(int(SAMPLES)):
        
        # Algorithm 2 - Line 4: Sample state according to current estimates
        state_idx = env.states.index(state)
        
        next_states_idxs = np.where(env.T[state_idx, :] > 0)[0].tolist()
        next_states = [env.states[i] for i in next_states_idxs]

        next_states_values = list(map(f_aux, next_states))

        pi = next_states_values / np.sum(next_states_values)

        next_state_idx = np.random.choice(next_states_idxs, p=pi)
        next_state = env.states[next_state_idx]

        (alpha1, alpha2, alpha3) = LRS.get_learning_rates()

        # Algorithm 2 - Line 5: Update subtasks' value functions
        if state not in env.G:

            if state in env.TS:
                break

            abs_ns = env.project_state_in_partition(next_state, partition)
            abs_s_idx, abs_ns_idx = env.abs_states.index(abs_state), env.abs_states.index(abs_ns)
            
            r = -env.abs_R[:, abs_s_idx] - np.log(gamma)

            next_abs_idxs = np.where(env.abs_T[abs_s_idx, :])[0].tolist()

            target = np.exp(r) * np.nanmean(Zs[:, next_abs_idxs], axis=1)
            
            Zs[:, env.abs_states.index(abs_state)] += alpha2 * \
                (target - Zs[:, env.abs_states.index(abs_state)])
            
        
        # (aux) make the transition in the LMDP
        (_, partition, abs_state), reward, _, _ = env.step(next_state)

        # Algorithm 2 - Line 6: Update estimated gamma 
        next_state_value = get_composed_value(next_state, env, Zs, exit_estimates)
        state_value = get_composed_value(state, env, Zs, exit_estimates)

        isw = (1 / len(next_states)) / pi[next_states.index(next_state)]

        deltaG = (
            isw * (np.exp(-reward) * next_state_value) / state_value - gamma)

        gamma += alpha3 * deltaG

        # Algorithm 2 - Lines 7 - 8: If 
        if state in env.exit_states and state != REF_STATE:
            e_idx = env.exit_states.index(state)
            deltaZ = (get_composed_value(next_state, env, Zs, exit_estimates) - exit_estimates[e_idx])

            exit_estimates[e_idx] += alpha1 * deltaZ

        state = next_state


        LRS.update()

        if i % 100 == 0:

            ERROR_GAMMA = np.abs(GAMMA_OPT - gamma)
            ERROR_Zs = np.abs(Zs - ZS_OPT).mean()
            ERROR_Z  = np.mean(Z_OPT[exit_states_idxs] - exit_estimates)

            log_dict = {"train/gamma": gamma,
                        "train/MAE_subtasks": ERROR_Zs,
                        "train/MAE_exit_states": ERROR_Z,
                        "train/Error_Gamma": ERROR_GAMMA,
                        "train/gt_gamma": GAMMA_OPT,
                        "learning_rates/z": alpha1,
                        "learning_rates/subtasks": alpha2,
                        "learning_rates/gamma": alpha3,
                        "step": i}
            
            wandb.log(log_dict)
            

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project,
        group="halmdps", 
        tags=["h-online"],)

    env_name = cfg.env_name
    env = gym.make(env_name)

    LRS = LearningRateScheduler(**cfg.lrs)

    algorithm(env, LRS)

    wandb.finish()

if __name__ == "__main__":

    main()