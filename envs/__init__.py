import gym

# ANCHOR: Office environments

gym.envs.register(
    id='NRoom-v0',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'NUM_ROOMS': (2, 1),
             'DIM': 3,
             'GOAL_POS_INSIDE_ROOM': (1,1),
             'GOAL_ROOMS': [(0, 0)],
             'PROBLEM_ID': 'lmdp-nroom-mini'
             },
)

gym.envs.register(
    id='NRoom-v1',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 3,
             'NUM_ROOMS': (2, 2),
             'GOAL_ROOMS': [(1, 1)],
             'GOAL_POS_INSIDE_ROOM': (1,1),
             'PROBLEM_ID': 'lmdp-nroom-1'
             },
)

gym.envs.register(
    id='NRoom-v2',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 3,
             'NUM_ROOMS': (2, 2),
             'GOAL_ROOMS': [(0, 0), (1, 1)],
             'GOAL_POS_INSIDE_ROOM': (1, 1),
             'PROBLEM_ID': 'lmdp-nroom-2'
             },
)

gym.envs.register(
    id='NRoom-v3',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 5,
             'NUM_ROOMS': (2, 2),
             'GOAL_ROOMS': [(1, 1)],
             'GOAL_POS_INSIDE_ROOM': (2, 3),
             'PROBLEM_ID': 'lmdp-nroom-3'
             },
)

gym.envs.register(
    id='NRoom-v4',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 5,
             'NUM_ROOMS': (2, 2),
             'GOAL_ROOMS': [(0, 0), (1, 1)],
             'GOAL_POS_INSIDE_ROOM': (2, 3),
             'PROBLEM_ID': 'lmdp-nroom-4'
             },
)

gym.envs.register(
    id='NRoom-v6',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 5,
             'NUM_ROOMS': (5, 5),
             'GOAL_ROOMS': [(0, 0), (1, 1), (0, 4), (4, 0), (4, 1)],
             'GOAL_POS_INSIDE_ROOM': (2, 3),
             'PROBLEM_ID': 'lmdp-nroom-6'
             },
)

gym.envs.register(
    id='NRoom-v7',
    entry_point='envs.NRoomDomainLMDP:NRoomEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 9,
             'NUM_ROOMS': (2, 2),
             'GOAL_ROOMS': [(0, 0), (1, 0), (0, 1), (1, 1)],
             'GOAL_POS_INSIDE_ROOM': (2, 3),
             'PROBLEM_ID': 'lmdp-nroom-7'
             },
)

gym.envs.register(
    id='Taxi-v0',
    entry_point='envs.TaxiDomainLMDP:TaxiEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 3,
             'PROBLEM_ID': 'lmdp-taxi-3'})


gym.envs.register(
    id='Taxi-v1',
    entry_point='envs.TaxiDomainLMDP:TaxiEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 5,
             'PROBLEM_ID': 'lmdp-taxi-5'})

gym.envs.register(
    id='Taxi-v2',
    entry_point='envs.TaxiDomainLMDP:TaxiEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 8,
             'PROBLEM_ID': 'lmdp-taxi-8'})

gym.envs.register(
    id='Taxi-v3',
    entry_point='envs.TaxiDomainLMDP:TaxiEnvLMDP',
    max_episode_steps=1000,
    kwargs={
             'DIM': 10,
             'PROBLEM_ID': 'lmdp-taxi-10'})