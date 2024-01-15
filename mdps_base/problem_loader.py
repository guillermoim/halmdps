from .nroom_mdp import create_flat_mdp as create_flat_nroom
from .taxi_mdp import create_flat_mdp as create_flat_taxi


def list_problems():
    return [
        "lmdp-nroom-mini",
        "lmdp-nroom-1",
        "lmdp-nroom-2",
        "lmdp-nroom-3",
        "lmdp-nroom-4",
        "lmdp-nroom-5",
        "lmdp-nroom-6",
        "lmdp-taxi-3",
        "lmdp-taxi-5",
        "lmdp-taxi-8",
        "lmdp-taxi-10"
    ]


def load_problem(problem: str):

    if problem == "lmdp-nroom-mini":
        return create_nroom_almdp_mini()
    elif problem == "lmdp-nroom-1":
        return create_nroom_almdp1()
    elif problem == "lmdp-nroom-2":
        return create_nroom_almdp2()
    elif problem == "lmdp-nroom-3":
        return create_nroom_almdp3()
    elif problem == "lmdp-nroom-4":
        return create_nroom_almdp4()
    elif problem == "lmdp-nroom-5":
        return create_nroom_almdp5()
    elif problem == "lmdp-nroom-6":
        return create_nroom_almdp6()
    elif problem == "lmdp-nroom-7":
        return create_nroom_almdp7()
    elif problem == "lmdp-nroom-8":
        return create_nroom_almdp8()
    elif problem == "lmdp-nroom-9":
        return create_nroom_almdp9()
    elif problem == "lmdp-taxi-3":
        return create_taxi_almdp3()
    elif problem == "lmdp-taxi-5":
        return create_taxi_almdp5()
    elif problem == "lmdp-taxi-8":
        return create_taxi_almdp8()
    elif problem == "lmdp-taxi-10":
        return create_taxi_almdp10()
    else:
        raise NotImplementedError(f"Problem {problem} not registered.",)


def create_nroom_almdp_mini():
    """
        Create a 2x1 rooms MDP, with room length 3, goal states at (1, 1, 1) goal room is (0,0)
    """

    problem = {}

    problem["ROOM_SIZE"] = 3
    problem["NUM_ROOMS"] = (2, 1)
    problem["GOAL_ROOMS"] = [(0, 0)]
    problem["GOAL_POS_INSIDE_ROOM"] = (1, 1)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(2e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e4)

    return LMDP, problem


def create_nroom_almdp1():
    """
        Create a 2x2 rooms MDP, with room length 3, goal states at (1, 1, 1) goal room is (0,0)
    """

    problem = {}

    problem["ROOM_SIZE"] = 3
    problem["NUM_ROOMS"] = (2, 2)
    problem["GOAL_ROOMS"] = [(1, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (1, 1)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(1e5)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e4)

    return LMDP, problem


def create_nroom_almdp2():
    """
        Create a 1x1 rooms MDP, with room length 3, goal states at (1, 1, 1) goal room are (0,0) and (1,1)
    """
    problem = {}

    problem["ROOM_SIZE"] = 3
    problem["NUM_ROOMS"] = (2, 2)
    problem["GOAL_ROOMS"] = [(0, 0), (1, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (1, 1)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(1e5)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e4)

    return LMDP, problem


def create_nroom_almdp3():
    """
        Create a 2x2 rooms MDP, with room length 5, goal states at (1, 2, 3) goal room is (1,1)
    """

    problem = {}

    problem["ROOM_SIZE"] = 5
    problem["NUM_ROOMS"] = (2, 2)
    problem["GOAL_ROOMS"] = [(1, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (2, 3)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(2e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e5)

    return LMDP, problem


def create_nroom_almdp4():
    """
        Create a 2x2 rooms MDP, with room length 5, goal states at (1, 2, 3) goal room is (0,0) and (1,1)
    """

    problem = {}

    problem["ROOM_SIZE"] = 5
    problem["NUM_ROOMS"] = (2, 2)
    problem["GOAL_ROOMS"] = [(0, 0), (1, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (2, 3)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(2e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e5)

    return LMDP, problem


def create_nroom_almdp5():
    """
        # TODO: Define this
        Create a 5x5 rooms MDP, with room length 5, goal states at (1, 2, 3) goal room is (1,1)
    """

    problem = {}

    problem["ROOM_SIZE"] = 5
    problem["NUM_ROOMS"] = (5, 5)
    problem["GOAL_ROOMS"] = [(1, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (2, 3)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(4e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e5)

    return LMDP, problem


def create_nroom_almdp6():
    """
        Create a 2x2 rooms MDP, with room length 5, goal states at (1, 2, 3) goal room is (1,1)
    """

    problem = {}

    problem["ROOM_SIZE"] = 5
    problem["NUM_ROOMS"] = (5, 5)
    problem["GOAL_ROOMS"] = [(0, 0), (1, 1), (0, 4), (4, 0), (4, 1)]
    problem["GOAL_POS_INSIDE_ROOM"] = (2, 3)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(4e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e5)

    return LMDP, problem


def create_nroom_almdp7():

    # TODO: Define this problem
    """
        Create a 3x3 rooms MDP, with room length 5, and goal states in rooms [(2,0), (0, 2), (2, 0)]
    """

    problem = {}

    problem["ROOM_SIZE"] = 5
    problem["NUM_ROOMS"] = (3, 3)
    problem["GOAL_ROOMS"] = [(0,0), (0, 2), (2, 0)]
    problem["GOAL_POS_INSIDE_ROOM"] = (2, 3)

    LMDP = create_flat_nroom(**problem)

    problem["REF_STATE"] = (0, 0, 0)
    problem["NUM_SAMPLES_ONLINE"] = int(1e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e6)

    return LMDP, problem


def create_taxi_almdp3():
    """
                Create an average reward Taxi LMDP with a grid of 3x3
    """
    DIM = 3
    problem = {}
    problem['DIM'] = DIM
    LMDP = create_flat_taxi(DIM)

    problem["REF_STATE"] = ((0, 0), (0, DIM - 1), (DIM - 1, 0))
    problem["NUM_SAMPLES_ONLINE"] = int(1e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(1e4)

    return LMDP, problem


def create_taxi_almdp5():
    """
                Create an average reward Taxi LMDP with a grid of 5x5
    """

    DIM = 5
    problem = {}
    problem['DIM'] = DIM
    LMDP = create_flat_taxi(DIM)

    problem["REF_STATE"] = ((0, 0), (0, DIM - 1), (DIM - 1, 0))
    problem["NUM_SAMPLES_ONLINE"] = int(1e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] = int(5e5)

    return LMDP, problem


def create_taxi_almdp8():
    """
                Create an average reward Taxi LMDP with a grid of 8x8
    """

    DIM = 8
    problem = {}
    problem['DIM'] = DIM
    LMDP = create_flat_taxi(DIM)

    problem["REF_STATE"] = ((0, 0), (0, DIM - 1), (DIM - 1, 0))
    problem["NUM_SAMPLES_ONLINE"] = int(1e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] =  int(5e5)

    return LMDP, problem


def create_taxi_almdp10():
    """
        Create an average reward Taxi LMDP with a grid of 10x10 
    """

    DIM = 10
    problem = {}
    problem['DIM'] = DIM
    LMDP = create_flat_taxi(DIM)

    problem["REF_STATE"] = ((0, 0), (0, DIM - 1), (DIM - 1, 0))
    problem["NUM_SAMPLES_ONLINE"] = int(2e6)
    problem["NUM_SAMPLES_HIERARCHICAL"] =  int(5e5)

    return LMDP, problem
