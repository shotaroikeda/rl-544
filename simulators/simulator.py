### General Class to abstract over different interfaces
from __future__ import print_function, absolute_import, division


class Simulator(object):
    def __init__(self, num_actions, state_shape, n_sims=1, multi_sim=False, reward_shape=False):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.multi_sim = multi_sim
        self.n_sims = n_sims
        self.reward_shape = reward_shape

    def random_action(self):
        """
        Do a random action.
        Return state, reward, terminial, action taken
        """
        raise NotImplementedError, "Subclass must implement random_action"

    def step(self, *action):
        """
        Do an action.
        Return state, reward, terminial, action_taken
        """
        raise NotImplementedError, "Subclass must implement step"

    def close(self):
        """
        Close the simulator
        """
        raise NotImplementedError, "Subclass must implement close"
