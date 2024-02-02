import argparse
import gym 
import envs
import pickle as pkl

import numpy as np

from algorithms.flat_online_algorithms import DifferentialLogTDLearning

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--eta", type=float, default=1)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    eta = args.eta

    with open(f"results/ground_truth/{env.unwrapped.spec.id}_{eta:.4f}.pkl", "rb") as fp:

       Z_OPT, GAMMA_OPT, _, _ =  pkl.load(fp)
    
    REF_STATE = env.G[0]

    algo = DifferentialLogTDLearning(env, decay_factor_gamma=.95, 
                                        decay_factor_z=.99,
                                        decay_step_gamma=5000,
                                        decay_step_z=10000,
                                        eta = eta
                                        )
    

    (state, _, _) = env.reset()
    
    for i in range((int(1e6))):

        # Get the current state and next states' values.
        next_states_idxs = np.where(env.T[env.states.index(state), :] > 0)[0].tolist()
        # next_states = list(map(lambda x: env.states[x], next_states))
        next_states_values = algo.z[next_states_idxs]
        pi = next_states_values / np.sum(next_states_values)

        # print(state, pi, env.states[next_states_idxs[np.argmax(pi)]])
        try:
            action = np.random.choice(len(next_states_idxs), p=pi)
        except:
            print(state, [env.states[i] for i in next_states_idxs])
            print(next_states_values)
            exit()

        next_state = env.states[next_states_idxs[action]]

        (next_state, _, _), reward, _, _ = env.step(next_state)
        algo.samples+=1

        isw = (1 / len(next_states_idxs)) / pi[action]

        algo.update(state=state, next_state=next_state, reward=reward, isw=isw, update_vf=state != REF_STATE)
        
        if i % 10000 == 0:
            pass
            print(np.abs(Z_OPT - algo.z).mean())
            # print(GAMMA_OPT, algo.gamma)
            # print(np.abs(np.log(Z_OPT[:len(env.IS)])/eta - np.log(algo.z[:len(env.IS)])/eta).mean())
            # print(algo.lr_g, algo.lr_vf)
            # print()

        state = next_state


    print(np.abs(Z_OPT - algo.z))
    print(Z_OPT)



