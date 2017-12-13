######################################################################
# Replay Memory
# Stores state, action, next_state, reward, done
import torch
from torch.autograd import Variable
import numpy as np


class ReplayMemory(object):
    def __init__(self, state_shape, capacity=20000):
        ''' Initializes empty replay memory '''
        self.capacity = capacity
        self.memory = torch.zeros(*((capacity, ) + state_shape))
        self.idx = 0

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


class PrioritizedReplayMemory(object):

    def __init__(self, state_shape, n_sims, capacity=1 << 15, eps=1e-6, alpha=0.8, mode=torch):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.states = torch.zeros(*((capacity,) + state_shape))
        self.actions = torch.zeros(capacity).long()
        self.rwrds = torch.zeros(capacity)
        self.terms = torch.zeros(capacity).byte()
        self.state_shape = state_shape
        self.n_sims = n_sims

        self.position = 0
        self.eps = eps
        self.alpha = alpha

        self.mode = mode

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def _add(self, p, state, rwrd, act, term):
        idx = self.position + self.capacity - 1

        self.states[self.position] = state
        self.rwrds[self.position] = rwrd
        self.actions[self.position] = act
        self.terms[self.position] = term
        self._update(idx, p)

        self.position = (self.position + 1) % self.capacity

    def _update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _quit(self):
        self.terms[(self.position - 1) % self.capacity] = True

    def total(self):
        return self.tree[0]

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        if self.terms[dataIdx]:
            return (idx,
                    self.tree[idx],
                    self.states[dataIdx],
                    self.rwrds[dataIdx],
                    self.actions[dataIdx],
                    self.terms[dataIdx],
                    torch.zeros(*self.state_shape))
        else:
            return (idx,
                    self.tree[idx],
                    self.states[dataIdx],
                    self.rwrds[dataIdx],
                    self.actions[dataIdx],
                    self.terms[dataIdx],
                    self.states[(dataIdx + self.n_sims) % self.capacity])

    def _to_torch(self, arr):
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        else:
            return arr

    ## Single functions
    def add(self, loss, state, rwrd, act, term):
        pri = pow(loss + self.eps, self.alpha)
        state = self._to_torch(state)
        self._add(pri, state, rwrd, act, term)

    ### Batch functions ###
    def get_batch(self, batch_size=64):
        p = torch.rand(batch_size) * self.total()
        batch_idx, batch_sum, batch_states, batch_rwrds, batch_acts, batch_terms, batch_p_states = zip(
            *[self.get(s) for s in p])

        batch_states = Variable(torch.cat(batch_states, 0).view(
            *((batch_size,) + self.state_shape))).type(self.mode.FloatTensor)
        batch_rwrds = Variable(torch.Tensor(batch_rwrds).type(self.mode.FloatTensor)).unsqueeze(1)
        batch_acts = Variable(torch.Tensor(batch_acts).type(self.mode.LongTensor)).unsqueeze(1)
        batch_terms = Variable(torch.Tensor(batch_terms).type(self.mode.ByteTensor)).unsqueeze(1)
        batch_p_states = Variable(torch.cat(batch_p_states, 0).view(
            *((batch_size,) + self.state_shape))).type(self.mode.FloatTensor)

        return batch_idx, batch_sum, batch_states, batch_rwrds, batch_acts, batch_terms, batch_p_states

    def add_batch(self, losses, states, rwrds, actions, terms):
        priorities = torch.pow(losses + self.eps, self.alpha)
        # Flatten things that should be flat
        priorities = priorities.view(-1)
        losses = losses.view(-1)
        rwrds = rwrds.view(-1)
        actions = actions.view(-1)
        terms = terms.view(-1)

        for i in xrange(len(priorities)):
            self._add(priorities[i], states[i], rwrds[i], actions[i], terms[i])

    def update_batch(self, idxs, losses):
        priorities = torch.pow(losses + self.eps, self.alpha)
        priorities = priorities.view(-1)

        for i in xrange(len(priorities)):
            self._update(idxs[i], priorities[i])
