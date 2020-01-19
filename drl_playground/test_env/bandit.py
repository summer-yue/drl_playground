import gym
from gym import spaces
import numpy as np


class BanditEnv(gym.Env):
    """
    Simple two-armed Bandit environment for testing various learning
    algorithms on CartPole. A sensible algorithm should be able to solve this
    environment.

    """

    def __init__(self):
        """
        Initialize the environment. Both the observation space and the
        action space are set to the dimensions for CartPole-v0.

        TODO: pass in action space and observation space as params. The
            algorithm should work on any shape of environments.

        """
        self.observation_space = spaces.Box(
            low=np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01,
                          -3.4028235e+38]),
            high=np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01,
                           3.4028235e+38]))
        self.action_space = spaces.Discrete(2)

        # We record the time from start to decide whether we terminate the
        # environment after self.max_time_steps.
        self.time_from_start = 0
        self.max_time_steps = 5

        # self.observation is meaningless as the reward never depends on the
        # current observation.
        self.observation = np.zeros((4,))
        super().__init__()

    def step(self, action):
        """
        We give an reward if action == 1, independent of the observations.
        The game terminates after self.max_time_steps.

        Args:
            action: an integer representing an action, either 1 or -1.

        Returns:
            observation: always None. Needed to fit the gym signature.
            reward: the reward from this action.
            done: whether or not the environment has terminated.
            info: a string containing helpful information such as the current
                time step.
        """
        self.time_from_start += 1

        if action == 1:
            reward = 1.0
        else:
            reward = 0.0
        # Game always terminates after self.max_time_steps time steps.
        done = self.time_from_start >= self.max_time_steps
        info = "Game time step: {} out of {}".format(self.time_from_start,
                                                     self.max_time_steps)
        return self.observation, reward, done, info

    def reset(self):
        """
        Reset the environment, because for the Bandit problem,
        the observation is meaningless, this function is a no-op.

        Returns:
            observation: the initial state of the environment, always
            [0., 0., 0., 0.].

        """
        self.time_from_start = 0
        return self.observation

    def render(self, mode='human'):
        """
        Rendering for the simple Bandit is not supported.

        """
        raise NotImplementedError
