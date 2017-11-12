import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'nonterminal'))

######################################################################
# Replay Memory
# Stores state, action, next_state, reward, done

class ReplayMemory():
    def __init__(self, capacity = 10000):
        ''' Initializes empty replay memory '''
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        ''' Stores item in replay memory '''
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ''' Samples item from replay memory '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' Current size or replay memory '''
        return len(self.memory)

######################################################################
# Deep Q-Network
# Defines DQN for cartpole

class DQN(nn.Module):
    '''
    Define a fully-connected neural network
    '''

    def __init__(self, input_size, output_size, hidden_sizes,
                    hidden_activation = F.relu):
        '''
        Instantiates DQN

        Position Arguments:
        input_size : (int) size of state tuple
        output_size : (int) size of discrete actions
        hidden_sizes : (list of ints) sizes of hidden layer outputs

        Keyword Arguments:
        hidden_activation : (torch functional) nonlinear activations
        '''
        # TODO: Define network architecture.
        # Hint: Look at documentation for nn.ModuleList() and nn.Linear
        pass

    def forward(self, x):
        '''
        Given batch of states outputs Q(s, a) for states s in batch for all
        actions a.

        Positional Arguments:
        x : (Variable FloatTensor) tensor of states of size (batch_size, input_size)

        Return:
        q : (Variable FloatTensor) tensor of Q values of size (batch_size, output_size)
        '''
        # TODO: define the forward pass for your network
        pass

######################################################################
# Cart Pole Agent
# Defines agent for cartpole

class CartPoleAgent():
    '''
    Defines cartpole agent
    '''

    def __init__(self, num_episodes = 100, discount = 0.999, epsilon_max = 1.0,
                epsilon_min = 0.05, epsilon_decay = 200, lr = 1e-1,
                batch_size = 128, copy_frequency = 10):
        '''
        Instantiates DQN agent

        Keyword Arguments:
        num_episodes : (int) number of episodes to run agent
        discount: (float) discount factor (should be <= 1)
        epsilon_max : (float) initial epsilon. Epsilon controls how often agent selects
        random action given the state
        epsilon_min : (float) final epsilon
        epsilon_decay : (float) controls rate of epsilon decay.
        lr : (float) learning_rate
        batch_size : (int) size of batch sampled from replay memory.
        copy_frequency : (int) copy after a certain number of time steps
        '''
        # Save relevant hyperparameters
        self.num_episodes = num_episodes
        self.discount = discount
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.copy_frequency = copy_frequency

        # TODO : Instantiate replay memory, DQN, target DQN, optimizer, and gym environment
        self.env = gym.make('CartPole-v1')
        self.memory = ReplayMemory()

    def select_action(self, state, steps_done = 0, explore = True):
        '''
        Given state returns an action by either randomly choosing an action or
        choosing an action that maximizes the expected reward (Q(s, a)).

        Return:
        action : (int) action choosen from state
        '''
        sample = random.random()
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
            math.exp(-1. * steps_done / self.epsilon_decay)

        # Replace the following line below. This line selects a random action.
        return random.randint(0, 1)

        # TODO : with prob 1 - epsilon choose action a to maximize Q(state, a)
        # Hint : make sure you set the volatile flag. You want the model to
        # be in evaluation/inference mode

        # TODO: with prob epsilon choose action randomly. Actions are between 0 and 1

    def train(self, show_plot = True):
        '''
        Trains the cartpole agent.

        Keyword Arguments:
        show_plot : (boolean) indicates whether duration curve is plotted
        '''
        steps_done = 0
        durations = []
        for ep in range(self.num_episodes):
            state = self.env.reset()
            state = torch.FloatTensor([state])
            done = False
            duration = 0
            while not done:
                # Select action and take step
                action = self.select_action(state, steps_done)
                next_state, reward, done, _ = self.env.step(action)

                # Convert s, a, r, s', d to tensors
                next_state = torch.FloatTensor([next_state])
                action = torch.LongTensor([[action]])
                reward = torch.FloatTensor([reward])
                nonterminal = torch.ByteTensor([not done])

                # TODO: Remember s, a, r, s', d in replay memory

                steps_done += 1
                state = next_state
                duration += 1

                # Sample from replay memory if the current memory size is greater
                # than the batch size
                if len(self.memory) >= self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*batch))
                    state_batch = Variable(torch.cat(batch.state))
                    action_batch = Variable(torch.cat(batch.action))
                    next_state_batch = Variable(torch.cat(batch.next_state), volatile = True)
                    reward_batch = Variable(torch.cat(batch.reward))
                    nonterminal_mask = torch.cat(batch.nonterminal)

                    # TODO: Predict Q(s, a) for s in batch

                    # TODO: Calcuate target values
                    # if terminal state, then target = rewards
                    # else target = r(s, a) + discount * max_a Q(s', a) where s' is
                    # next state.
                    # (1) Make sure you are using the target network in this step,
                    # not the current network.
                    # (2) Make sure that the target network is not trained on,
                    # also make sure that the volatile flag in next state batch does
                    # not affect the final loss output.

                    # TODO: Calculate loss function and optimize.
                    # This teaches you show to use pytorch loss functions and optimizers

                # TODO: Copy currently trained network to target network
                # Most likely unneeded for cart pole, but targets networks are used
                # generally in DQN.

                # Plot durations
                if done and show_plot:
                    durations.append(duration)
                    self.plot_durations(durations)
                    duration = 0

    def plot_durations(self, durations):
        '''
        Plots duration curve

        Positional Arguments:
        durations : (list of ints) duration for every episode
        '''
        plt.figure(1)
        plt.clf()
        durations_a = np.array(durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_a)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def run_and_visualize(self):
        ''' Runs and visualizes the cartpole agents. '''
        # TODO: run and visualize agent
        # Hint this requires self.select_action
        pass


def main():
    cpa = CartPoleAgent()
    cpa.train()
    cpa.run_and_visualize()

if __name__ == '__main__':
    main()
