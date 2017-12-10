from __future__ import print_function

from memory.replay import PrioritizedReplayMemory
from networks.network import DQN, print_weight
from simulators.gym_simulator import CartPole

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from cust_torch.optim import LBFGS

import time

from tqdm import tqdm
import random
import numpy as np

import matplotlib.pyplot as plt
import visdom
import copy

CUDA = False


def pp_target(tar_val):
    return ((tar_val / 100.0) - 5.0)


def eps_decay(curr_step, decay_f=1):
    return 0.02 + (decay_f * 1500.0) / (decay_f * 1530.0 + curr_step)


class Trainer(object):
    def init_lbfgs(self):
        self.optim = torch.optim.LBFGS(self.network.parameters(), lr=self.lr,
                                       history_size=self.history_size, max_iter=1)

    def __init__(self, memory, network, simulator, TOptimClass, eps_decay_fn=eps_decay,
                 lr=0.01, history_size=100, gamma=0.9):
        assert issubclass(
            TOptimClass, torch.optim.Optimizer), "TOptimClass must be a torch optimizer"

        self.simulator = simulator
        self.memory = memory
        self.network = network
        self.lr = lr
        self.history_size = history_size

        if CUDA:
            self.network.cuda()

        self.mode = torch
        if CUDA:
            self.mode = torch.cuda

        if issubclass(TOptimClass, torch.optim.LBFGS):
            self.init_lbfgs()
            self.schedule = StepLR(self.optim, step_size=1000, gamma=0.5, last_epoch=-1)
        else:
            self.optim = TOptimClass(self.network.parameters(), lr=lr)
            self.schedule = StepLR(self.optim, step_size=1000, gamma=0.5, last_epoch=-1)

        self.steps = 0
        self.init_time = time.time()
        self.gamma = gamma

        self.eps_decay_fn = eps_decay_fn

        # NOTE: Set up target network
        self.use_target_net = True
        self._sync()

        # NOTE: Set up Visdom for plotting intermediary episodes
        self.vis = visdom.Visdom()
        self.vis.close()

    def _train_batch(self, batch_size=64):
        self.optim.zero_grad()

        idxs, sums, states, rwrds, acts, terms, p_states = self.memory.get_batch(
            batch_size=batch_size)
        preds = self.network.preds(states, acts)

        if self.use_target_net:
            net = self.target_net
        else:
            net = self.network

        targets = net.act(p_states)[0].unsqueeze(1)
        targets[terms] = 0
        targets.detach_()

        losses = self.network.loss(preds, rwrds + self.gamma * pp_target(targets))
        grad_loss = losses.mean()

        # NOTE: Insert Closure here
        def clip_gradient(model, clip):
            """Clip the gradient"""
            if clip is None:
                return
            for p in model.parameters():
                if p.grad is None:
                    continue
                else:
                    p.grad.data = p.grad.data.clamp(-clip, clip)

        net_backup = copy.deepcopy(self.network)

        def closure():
            self.optim.zero_grad()
            loss = self.network.loss(preds, rwrds + self.gamma * pp_target(targets)).mean()

            # l1_crit = nn.L1Loss(size_average=False)
            # reg_loss = 0
            # for param in self.network.parameters():
            #     reg_loss += l1_crit(param, Variable(torch.zeros(param.size())))

            # factor = 0.05
            # loss += factor * reg_loss

            if loss.data[0] > 1000:
                print('LOSS REALLY BIG')
                print("Preds:", preds)
                print('Targets: ', rwrds + self.gamma * pp_target(targets))
                print(loss.data[0])
                # with open('netw.log', 'w') as f:
                #     print_weight(self.network, f)
                # TOO_BIG = False
                # print("Target: ", rwrds + self.gamma * targets)
                # if self.use_target_net:
                #     self.network = copy.deepcopy(self.target_net)
                #     self.network.train()
                #     self.init_lbfgs()
                # else:
                #     exit(1)
            else:
                # print("Preds:", preds)
                pass
            loss.backward(retain_graph=True)
            # clip_gradient(self.network, 1)
            return loss

        if isinstance(self.optim, torch.optim.LBFGS):
            self.optim.step(closure)
        else:
            grad_loss.backward()
            self.optim.step()

        losses = losses.data.cpu()
        self.memory.update_batch(idxs, losses)

        return losses.mean()

    def _eval(self, num_evals=10):
        all_action_bin = np.zeros(self.simulator.num_actions)
        all_rewards = np.zeros(num_evals)
        all_pred_mean = np.zeros(num_evals)
        all_pred_std = np.zeros(num_evals)
        all_loss_mean = np.zeros(num_evals)
        all_loss_std = np.zeros(num_evals)
        all_episode_len = np.zeros(num_evals)

        self.network.eval()
        for i in range(num_evals):
            _action_list = []
            _rwrd_list = []
            _pred_list = []
            _loss_list = []
            episode_len = 0

            state = self.simulator.reset()
            state = Variable(torch.from_numpy(state).type(self.mode.FloatTensor),
                             volatile=True).unsqueeze(0)
            while True:
                pred, action = self.network.act(state)

                state, rwrd, term, _ = self.simulator.step(
                    action.data.cpu().numpy()[0], render=True)

                state = Variable(torch.from_numpy(state).type(self.mode.FloatTensor),
                                 volatile=True).unsqueeze(0)
                _pred = self.network.act(state)[0]

                loss = self.network.loss(
                    pred,
                    Variable(torch.Tensor([rwrd]).type(self.mode.FloatTensor),
                             volatile=True).unsqueeze(0) + self.gamma * pp_target(_pred)
                ).data.cpu().numpy()[0]

                action = action.data.cpu().numpy()[0]
                pred = pred.data.cpu().numpy()[0]

                _action_list.append(action)
                _rwrd_list.append(rwrd)
                _pred_list.append(pred)
                _loss_list.append(loss)
                episode_len += 1

                if term:
                    break

            all_action_bin += np.bincount(_action_list, minlength=self.simulator.num_actions)
            all_rewards[i] = np.sum(_rwrd_list)
            all_pred_mean[i] = np.mean(_pred_list)
            all_pred_std[i] = np.std(_pred_list)
            all_loss_mean[i] = np.mean(_loss_list)
            all_loss_std[i] = np.std(_loss_list)
            all_episode_len[i] = episode_len

        self.network.train()
        return (all_action_bin, all_rewards.mean(),
                all_pred_mean.mean(), all_pred_std.mean(),
                all_loss_mean.mean(), all_loss_std.mean(),
                all_episode_len.mean())

    def _step(self, eps):
        curr_state = self.simulator.get_state()
        curr_state = Variable(torch.from_numpy(curr_state).type(
            self.mode.FloatTensor), volatile=True).unsqueeze(0)

        if random.random() < eps:
            p_state, rwrd, term, action = self.simulator.random_action()

            action = Variable(torch.Tensor([action]).type(
                self.mode.LongTensor), volatile=True).unsqueeze(0)
            pred = self.network.preds(curr_state, action)

            action = action[0]
        else:
            pred, action = self.network.act(curr_state)
            p_state, rwrd, term, _ = self.simulator.step(action.data.cpu().numpy()[0])

        rwrd = Variable(torch.Tensor([rwrd]).type(
            self.mode.FloatTensor), volatile=True).unsqueeze(0)

        # For loss w/ disccount factor, we have to use the target net
        if term:
            state_val = 0
        else:
            p_state = Variable(torch.from_numpy(p_state).type(
                self.mode.FloatTensor), volatile=True).unsqueeze(0)
            if self.use_target_net:
                net = self.target_net
            else:
                net = self.network

            state_val = net.act(p_state)[0].unsqueeze(1)

        loss = self.network.loss(pred, rwrd + self.gamma * pp_target(state_val))

        return (loss.data.cpu()[0, 0],
                curr_state.data.cpu()[0],
                rwrd.data.cpu()[0, 0],
                action.data.cpu()[0],
                term)

    def _sync(self):
        """
        Sync the target network
        """
        if self.use_target_net:
            self.target_net = copy.deepcopy(self.network)

            if CUDA:
                self.target_net.cuda()

            self.target_net.eval()

            if isinstance(self.optim, torch.optim.LBFGS):
                self.init_lbfgs()

    def n_step(self, steps, eps):
        batch_loss, batch_state, batch_rwrd, batch_action, batch_term = zip(
            *[self._step(eps) for i in range(steps)])

        losses = torch.Tensor(batch_loss).float()
        states = torch.cat(batch_state, 0)
        rwrds = torch.Tensor(batch_rwrd).float()
        actions = torch.Tensor(batch_action).long()
        terms = torch.Tensor(batch_term).byte()

        self.memory.add_batch(losses, states, rwrds, actions, terms)

    def warmup(self):
        """
        Warm up experience
        """
        iters = self.memory.capacity

        for _ in tqdm(range(iters)):
            loss, state, rwrd, action, term = self._step(self.eps_decay_fn(self.steps))
            self.memory.add(loss, state, rwrd, action, term)

    def train(self, num_epochs=100000):
        self.warmup()

        all_rwrds = np.zeros(num_epochs)
        all_pred_means = np.zeros(num_epochs)
        all_pred_std = np.zeros(num_epochs)
        all_loss_means = np.zeros(num_epochs)
        all_loss_std = np.zeros(num_epochs)
        all_ep_len = np.zeros(num_epochs)

        for i in range(num_epochs):
            # NOTE: Book Keeping for evaluation
            act_bin, rwrd, pred_means, pred_std, loss_means, loss_std, ep_len = self._eval()
            all_rwrds[i] = rwrd
            all_pred_means[i] = pred_means
            all_pred_std[i] = pred_std
            all_loss_means[i] = loss_means
            all_loss_std[i] = loss_std
            all_ep_len[i] = ep_len

            # Plot Information
            self.vis.bar(X=act_bin,
                         opts={'title': 'Action Distribution Over Episodes'})
            if i == 0:
                self.rwrd_win = self.vis.line(X=np.arange(i + 1),
                                              Y=all_rwrds[:i + 1],
                                              opts={'title': 'Rewards Over Episodes'})
                self.pred_means_win = self.vis.line(X=np.arange(i + 1),
                                                    Y=all_pred_means[:i + 1],
                                                    opts={'title': 'Average Preds Over Episodes'})
                self.pred_std_win = self.vis.line(X=np.arange(i + 1),
                                                  Y=all_pred_std[:i + 1],
                                                  opts={'title': 'STD Preds Over Episodes'})
                self.loss_means_win = self.vis.line(X=np.arange(i + 1),
                                                    Y=all_loss_means[:i + 1],
                                                    opts={'title': 'Average Loss Over Episodes'})
                self.loss_std_win = self.vis.line(X=np.arange(i + 1),
                                                  Y=all_loss_std[:i + 1],
                                                  opts={'title': 'STD Loss Over Episodes'})
                self.ep_len_win = self.vis.line(X=np.arange(i + 1),
                                                Y=all_ep_len[:i + 1],
                                                opts={'title': 'Average Episode Length Over Episodes'})

            else:
                self.vis.line(X=np.arange(i + 1),
                              Y=all_rwrds[:i + 1],
                              opts={'title': 'Rewards Over Episodes'},
                              win=self.rwrd_win)
                self.vis.line(X=np.arange(i + 1),
                              Y=all_pred_means[:i + 1],
                              opts={'title': 'Average Preds Over Episodes'},
                              win=self.pred_means_win)
                self.vis.line(X=np.arange(i + 1),
                              Y=all_pred_std[:i + 1],
                              opts={'title': 'STD Preds Over Episodes'},
                              win=self.pred_std_win)
                self.vis.line(X=np.arange(i + 1),
                              Y=all_loss_means[:i + 1],
                              opts={'title': 'Average Loss Over Episodes'},
                              win=self.loss_means_win)
                self.vis.line(X=np.arange(i + 1),
                              Y=all_loss_std[:i + 1],
                              opts={'title': 'STD Loss Over Episodes'},
                              win=self.loss_std_win)
                self.vis.line(X=np.arange(i + 1),
                              Y=all_ep_len[:i + 1],
                              opts={'title': 'Average Episode Length Over Episodes'},
                              win=self.ep_len_win)

            # NOTE: Actual training
            self.schedule.step()
            for j in tqdm(range(600)):
                # Take Random Batch
                l = self._train_batch(64)
                # Increment Step
                self.steps += 1
                eps = self.eps_decay_fn(self.steps)

                # Add new experiences
                self.n_step(8, eps)
                if j % 10 == 0:
                    print('loss: %f' % (l))

                if (i * 600 + j) % 100 == 0:
                    print('---sync sync sync---')
                    self._sync()

            self.memory._quit()


if __name__ == '__main__':
    if CUDA:
        mode = torch.cuda
    else:
        mode = torch

    simulator = CartPole()
    memory = PrioritizedReplayMemory(simulator.state_shape, capacity=1 << 12, mode=mode, alpha=0.8)
    network = DQN(simulator.state_shape, simulator.num_actions, layers=1)

    if CUDA:
        network = network.cuda()

    # optimizer = torch.optim.Adam
    optimizer = torch.optim.LBFGS
    # optimizer = torch.optim.SGD

    if issubclass(optimizer, torch.optim.LBFGS):
        tr = Trainer(memory, network, simulator, optimizer,
                     lr=1.0, history_size=10, gamma=0.9)
    else:
        tr = Trainer(memory, network, simulator, optimizer)

    tr.train()
