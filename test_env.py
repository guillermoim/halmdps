import argparse
import gym 
import envs


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")

    args = parser.parse_args()

    env  = gym.make(args.env_name)

    # print(env.states.index(env.G))

    print(env.T[24, :])


    env.reset()

    print(env.step((0,0,1)))
    print(env.step((0,1,1)))
    print(env.step((1,1,1)))
    print(env.step((0,1,1)))