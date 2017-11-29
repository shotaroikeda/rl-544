from __future__ import print_function, absolute_import, division
from .simulator import Simulator

import gym
import random


# Might be cleaner to do, but not sure if necessary
class GymSimulator(Simulator):
    def __init__(self, env_name):
        pass


class CartPole(Simulator):
    def __init__(self):
        super(CartPole, self).__init__(2, (4,))
        self.env = gym.make("CartPole-v1")
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        return self.get_state()

    def random_action(self):
        action = random.sample([0, 1], 1)
        return self.step(action[0])

    def step(self, action, render=False):
        if render:
            print(action)
            self.env.render()

        state, rwrd, term, aux = self.env.step(action)
        self.state = state

        if term:
            self.reset()

        return state, rwrd, term, action

    def get_state(self):
        return self.state
