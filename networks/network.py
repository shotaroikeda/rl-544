import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def init_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight)

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Sequential):
            for mini_module in list(m.modules())[1:]:
                init_weight(mini_module)


def print_weight(net, f):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            f.write("%s\n" % (str(m.weight)))

            if m.bias is not None:
                f.write('%s\n' % (m.bias))
        elif isinstance(m, nn.Sequential):
            for mini_module in list(m.modules())[1:]:
                print_weight(mini_module, f)


class DQN(nn.Module):
    def __init__(self, state_shape, num_actions, layers=1, mode=torch):
        super(DQN, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(np.prod(state_shape), 512, bias=True),
            nn.ReLU())
        self.linear2 = nn.Sequential(
            *[nn.Sequential(nn.Linear(512, 512, bias=True),
                            nn.ReLU())
              for _ in range(layers)]
        )

        self.val = nn.Linear(512, 1, bias=True)  # No bias for output
        self.adv = nn.Linear(512, num_actions, bias=True)  # No bias for output

        init_weight(self)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        adv = self.adv(x)
        return adv
        # val = self.val(x)

        # x = adv - adv.mean(1).unsqueeze(1) + val
        # return x

    def loss(self, preds, rwrds):
        # x = torch.pow(preds - rwrds, 2)
        x = F.smooth_l1_loss(preds, rwrds, size_average=False).unsqueeze(1)
        return x

    def preds(self, states, action_idxs):
        x = self.forward(states)
        rng = torch.arange(0, states.size()[0]).type(torch.LongTensor)
        x = x[rng, action_idxs.squeeze(1).data].unsqueeze(1)
        return x

    def act(self, x):
        return torch.max(self.forward(x), 1)


class DDQN(nn.Module):
    def __init__(self, state_shape, num_actions, layers=1):
        super(DDQN, self).__init__()
        self.dqn1 = DQN(state_shape, num_actions, layers, mode)
        self.dqn2 = DQN(state_shape, num_actions, layers, mode)

        self.mode = mode

    def forward(self, x):
        return torch.cat([self.dqn1(x), self.dqn2(x)], 1).view(x.size()[0], 2, -1)

    def loss(self, preds, rwrds):
        x = torch.pow(preds - rwrds, 2)
        return x

    def preds(self, states, action_idxs):
        x = self.act(states)[0]
        rng = torch.arange(0, states.size()[0]).type(torch.LongTensor)
        x = x[rng, action_idxs.squeeze(1).data].unsqueeze(1)
        return x

    def act(self, x):
        select_idx = (torch.rand(x.size()[0]) > 0.5).type(self.mode.LongTensor)
        x = self.forward(x)
        x = x[torch.arange(x.size()[0]).type(self.mode.LongTensor),
              select_idx]
        return torch.max(x, 1)
