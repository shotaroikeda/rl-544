from __future__ import print_function, absolute_import, division
from .simulator import Simulator

import gym
import random
import numpy as np


# Might be cleaner to do, but not sure if necessary
class GymSimulator(Simulator):
    def __init__(self, env_name):
        pass


class CartPole(Simulator):
    def __init__(self, reward_shape=False):
        super(CartPole, self).__init__(2, (4,), reward_shape=reward_shape)
        self.env = gym.make("CartPole-v0")
        self.reset()
        self.frame_counter = 0

    def reset(self):
        self.state = self.env.reset()
        self.frame_counter = 0
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
            if self.reward_shape:
                if self.frame_counter > 199:
                    rwrd = self.frame_counter
                else:
                    rwrd = -self.frame_counter
            self.reset()

        self.frame_counter += 1

        return state, rwrd, term, action

    def get_state(self):
        return self.state

    def close(self):
        self.env.render(close=True)


class MultiCartPole(Simulator):
    def __init__(self, n_sims, reward_shape=False):
        super(MultiCartPole, self).__init__(2, (4,), n_sims=n_sims, multi_sim=True,
                                            reward_shape=reward_shape)
        self.simulators = [CartPole(reward_shape=reward_shape) for _ in range(n_sims)]

    def reset(self):
        states = [self.simulators[i].reset() for i in range(self.n_sims)]
        return np.asarray(states)

    def random_action(self):
        actions = [random.sample([0, 1], 1)[0] for _ in range(self.n_sims)]
        return self.step(actions)

    def step(self, action, render=False):
        states = [self.simulators[i].step(action[i], render) for i in range(self.n_sims)]
        return zip(*states)

    def get_state(self):
        states = [self.simulators[i].state for i in range(self.n_sims)]
        return np.asarray(states)

    def close(self):
        for i in range(self.n_sims):
            self.simulators[i].close()
