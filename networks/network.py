import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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
    def __init__(self, state_shape, num_actions, layers=0, mode=torch):
        super(DQN, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128, bias=True),
            nn.ReLU())
        self.linear2 = nn.Sequential(
            *[nn.Sequential(nn.Linear(64, 64, bias=True),
                            nn.ReLU())
              for _ in range(layers)]
        )

        self.val = nn.Linear(128, 1, bias=True)  # No bias for output
        self.adv = nn.Linear(128, num_actions, bias=True)  # No bias for output

        init_weight(self)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        adv = self.adv(x)
        return adv
        # val = self.val(x)

        # x = adv - adv.mean(1).unsqueeze(1) + val
        # return x

    def loss(self, preds, rwrds, delta=1.0):
        ###########
        # L2 Loss #
        ###########
        x = torch.pow(preds - rwrds, 2)

        ###########
        # L1 Loss #
        ###########
        # loss = preds - rwrds
        # x = 0.5 * torch.pow(loss, 2)

        # lin_x = delta * (torch.abs(loss) - 0.5 * delta)
        # x[torch.abs(loss) < delta] = lin_x
        return x

    def preds(self, states, action_idxs):
        x = self.forward(states)
        rng = torch.arange(0, states.size()[0]).type(torch.LongTensor)
        x = x[rng, action_idxs.squeeze(1).data].unsqueeze(1)
        return x

    def act(self, x):
        return torch.max(self.forward(x), 1)
