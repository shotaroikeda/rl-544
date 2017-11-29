### General Class to abstract over different interfaces
from __future__ import print_function, absolute_import, division


class Simulator(object):
    def __init__(self, num_actions, state_shape):
        self.num_actions = num_actions
        self.state_shape = state_shape

    def random_action(self):
        """
        Do a random action.
        Return state, reward, terminial, action taken
        """
        raise NotImplementedError, "Subclass must implement random_action"

    def step(self, action):
        """
        Do an action.
        Return state, reward, terminial, action_taken
        """
        raise NotImplementedError, "Subclass must implement step"
