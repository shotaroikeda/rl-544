from __future__ import print_function

from memory.replay import PrioritizedReplayMemory
from networks.network import DQN, print_weight
from simulators.gym_simulator import CartPole, MultiCartPole

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
    return tar_val
    # tar_val / 10  # (tar_val / 500.0) - 0.5
    if isinstance(tar_val, (Variable, torch.Tensor, np.ndarray)):
        tar_val[tar_val < -3] = -3
        tar_val[tar_val > 3] = 3
        return tar_val
    else:
        if tar_val < -3:
            return -3
        elif tar_val > 3:
            return 3
        else:
            return tar_val


def pp_rwrd(rwrd):
    # return rwrd / 500 - 0.5
    return rwrd


def eps_decay(curr_step, decay_f=10, debug=False):
    if debug:
        print(curr_step)
    eps = 0.02 + (decay_f * 1500.0) / (decay_f * 1530.0 + curr_step)
    if debug:
        print(eps)
    return eps


def eps_decay_lin(curr_step, debug=False):
    eps = max(0.98 - 0.000005 * curr_step, 0.02)
    if debug:
        print(curr_step)
        print(eps)
    return eps


class Trainer(object):
    def init_lbfgs(self):
        self.optim = torch.optim.LBFGS(self.network.parameters(), lr=self.lr,
                                       history_size=self.history_size, max_iter=1)

    def __init__(self, memory, network, simulator, eval_simulator, TOptimClass, eps_decay_fn=eps_decay,
                 lr=0.01, history_size=100, gamma=0.8):
        assert issubclass(
            TOptimClass, torch.optim.Optimizer), "TOptimClass must be a torch optimizer"

        self.simulator = simulator
        self.eval_simulator = eval_simulator
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
            self.schedule = StepLR(self.optim, step_size=1000000, gamma=0.5, last_epoch=-1)
        else:
            self.optim = TOptimClass(self.network.parameters(), lr=lr, eps=1e-4)
            self.schedule = StepLR(self.optim, step_size=15000, gamma=0.1, last_epoch=-1)

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

        losses = self.network.loss(preds, pp_rwrd(rwrds) + self.gamma * pp_target(targets))
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

        def closure():
            self.optim.zero_grad()
            loss = self.network.loss(preds, pp_rwrd(
                rwrds) + self.gamma * pp_target(targets)).mean()
            loss.backward(retain_graph=True)
            return loss

        if isinstance(self.optim, torch.optim.LBFGS):
            self.optim.step(closure)
        else:
            grad_loss.backward()
            # clip_gradient(self.network, 1)
            self.optim.step()

        losses = losses.data.cpu()
        self.memory.update_batch(idxs, losses)

        return losses.mean()

    def _eval(self, num_evals=100, render=False):
        all_action_bin = np.zeros(self.simulator.num_actions)
        all_rewards = np.zeros(num_evals)
        all_pred_mean = np.zeros(num_evals)
        all_pred_std = np.zeros(num_evals)
        all_loss_mean = np.zeros(num_evals)
        all_loss_std = np.zeros(num_evals)
        all_episode_len = np.zeros(num_evals)

        self.network.eval()
        for i in tqdm(range(num_evals)):
            _action_list = []
            _rwrd_list = []
            _pred_list = []
            _loss_list = []
            episode_len = 0

            state = self.eval_simulator.reset()
            state = Variable(torch.from_numpy(state).type(self.mode.FloatTensor),
                             volatile=True).unsqueeze(0)
            while True:
                pred, action = self.network.act(state)
                state, rwrd, term, _ = self.eval_simulator.step(
                    action.data.cpu().numpy()[0], render=render)

                state = Variable(torch.from_numpy(state).type(self.mode.FloatTensor),
                                 volatile=True).unsqueeze(0)
                _pred = self.network.act(state)[0]

                loss = self.network.loss(
                    pred,
                    Variable(torch.Tensor([pp_rwrd(rwrd)]).type(self.mode.FloatTensor),
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

            all_action_bin += np.bincount(_action_list, minlength=self.eval_simulator.num_actions)
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
            self.mode.FloatTensor), volatile=True)
        if not self.simulator.multi_sim:
            curr_state = curr_state.unsqueeze(0)

        if random.random() < eps:
            p_state, rwrd, term, action = self.simulator.random_action()

            action = Variable(torch.Tensor([action]).type(
                self.mode.LongTensor), volatile=True)
            if not self.simulator.multi_sim:
                action = action.unsqueeze(0)

            pred = self.network.preds(curr_state, action)
            action = action[0]
        else:
            pred, action = self.network.act(curr_state)
            if self.simulator.multi_sim:
                pred = pred.unsqueeze(0).unsqueeze(0)
                p_state, rwrd, term, _ = self.simulator.step(action.data.cpu().numpy())
            else:
                p_state, rwrd, term, _ = self.simulator.step(action.data.cpu().numpy()[0])

        rwrd = Variable(torch.Tensor([pp_rwrd(rwrd)]).type(
            self.mode.FloatTensor), volatile=True)

        if not self.simulator.multi_sim:
            rwrd = rwrd.unsqueeze(0)

        # For loss w/ disccount factor, we have to use the target net
        if self.simulator.multi_sim:
            p_state = Variable(torch.Tensor(p_state).float(), volatile=True)

            if self.use_target_net:
                net = self.target_net
            else:
                net = self.network

            state_val = net.act(p_state)[0]
            state_val[torch.Tensor(term).byte()] = 0
        else:
            if term and not isinstance(term, tuple):
                state_val = 0
            else:
                p_state = Variable(torch.from_numpy(p_state).type(
                    self.mode.FloatTensor), volatile=True)
                if not self.simulator.multi_sim:
                    p_state = p_state.unsqueeze(0)

                if self.use_target_net:
                    net = self.target_net
                else:
                    net = self.network

                state_val = net.act(p_state)[0].unsqueeze(1)

        q_val = pp_rwrd(rwrd) + self.gamma * pp_target(state_val)
        # print('state val', state_val)
        # print('preds', pred)
        # print('qval', q_val)
        # print('curr_state', curr_state)

        loss = self.network.loss(pred, q_val)

        if self.simulator.multi_sim:
            return (loss.data.cpu()[0, 0],
                    curr_state.data.cpu(),
                    rwrd.data.cpu()[0],
                    action.data.cpu(),
                    term)
        else:
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

        if self.simulator.multi_sim:
            losses = torch.cat(batch_loss, 0)
            states = torch.cat(batch_state, 0)
            rwrds = torch.cat(batch_rwrd, 0)
            actions = torch.cat(batch_action, 0)
            terms = torch.Tensor(batch_term).squeeze(0).byte()
        else:
            losses = torch.Tensor(batch_loss).float()
            states = torch.cat(batch_state, 0)
            rwrds = torch.Tensor(batch_rwrd).float()
            actions = torch.Tensor(batch_action).long()
            terms = torch.Tensor(batch_term).byte()

        losses = losses.view(steps * self.simulator.n_sims, -1)
        states = states.view(steps * self.simulator.n_sims, -1)
        rwrds = rwrds.view(steps * self.simulator.n_sims, -1)
        actions = actions.view(steps * self.simulator.n_sims, -1)
        terms = terms.view(steps * self.simulator.n_sims, -1)

        self.memory.add_batch(losses, states, rwrds, actions, terms)

    def warmup(self):
        """
        Warm up experience
        """
        iters = self.memory.capacity

        for _ in tqdm(range(iters / self.simulator.n_sims)):
            self.n_step(1, self.eps_decay_fn(self.steps))

    def train(self, num_epochs=30):
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

            eps = self.eps_decay_fn(self.steps, debug=True)

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
            for j in tqdm(range(1000)):
                self.schedule.step()
                # Take Random Batch
                l = self._train_batch(256)
                # Increment Step
                self.steps += 1
                eps = self.eps_decay_fn(self.steps, debug=False)

                # Add new experiences
                self.n_step(1, eps)
                if self.steps % 100 == 0:
                    print('loss: %f' % (l))

                if self.steps % 1000 == 0:
                    print('---sync sync sync---')
                    self._sync()

        return all_rwrds

    def close(self):
        self.simulator.close()
        self.eval_simulator.close()


def main(num_runs=5):
    results = []

    p_alpha = 0.0
    lr = 0.025
    history_size = 10
    gamma = 0.99
    n_simulators = 8

    for _ in range(num_runs):
        if CUDA:
            mode = torch.cuda
        else:
            mode = torch

        eval_simulator = CartPole(reward_shape=False)
        # simulator = CartPole(reward_shape=True)
        simulator = MultiCartPole(n_simulators, reward_shape=False)
        memory = PrioritizedReplayMemory(
            simulator.state_shape, simulator.n_sims, capacity=1 << 16, mode=mode, alpha=p_alpha)
        network = DQN(simulator.state_shape, simulator.num_actions, layers=0)

        if CUDA:
            network = network.cuda()

        # optimizer = torch.optim.Adam
        optimizer = torch.optim.LBFGS
        # optimizer = torch.optim.SGD

        if issubclass(optimizer, torch.optim.LBFGS):
            tr = Trainer(memory, network, simulator, eval_simulator,
                         optimizer, lr=lr, history_size=10, gamma=gamma,
                         eps_decay_fn=eps_decay_lin)
        else:
            tr = Trainer(memory, network, simulator, eval_simulator,
                         optimizer, lr=lr, gamma=gamma,
                         eps_decay_fn=eps_decay_lin)

        result = tr.train()
        results.append(result)

        tr.close()

    results = np.array(results)
    mean_rewards = results.mean(0)
    var_rewards = results.std(0)

    img_nm = ''

    if issubclass(optimizer, torch.optim.LBFGS):
        plt.title('Rewards for LBFGS $\\alpha={:.2f},\\eta={:.2f},h={:d}$'.format(
            p_alpha, lr, history_size))
        img_nm = 'lbfgs_{:.2f}_{:.2f}_{:d}.png'.format(p_alpha, lr, history_size)
    else:
        plt.title('Rewards for ADAM $\\alpha={:.2f},\\eta={:.2f}$'.format(p_alpha, lr))
        img_nm = 'adam_{:.2f}_{:.2f}.png'.format(p_alpha, lr)

    x = np.arange(mean_rewards.shape[0])
    plt.plot(x, mean_rewards)
    plt.fill_between(x, mean_rewards - var_rewards, mean_rewards + var_rewards, alpha=0.2)
    plt.savefig('report/assets/%s' % (img_nm), dpi=150)

    np.save('report/assets/%s.npy' % (img_nm), results)


if __name__ == '__main__':
    main()
