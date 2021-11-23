from __future__ import division

import torch.nn as nn
from .basemodel import BaseModel
from .tcn import TemporalConvNet
import torch.nn.functional as F
import torch


def NDweights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator_D1(nn.Module):
    def __init__(self):
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(NDweights_init)

    def forward(self, x):
        hidden = self.lrelu(self.fc1(x))
        h = self.fc2(hidden)
        return h


class EMLV2(BaseModel):
    def __init__(self, args):
        super(EMLV2, self).__init__(args)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        discriminator = Discriminator_D1()
        self.netD = discriminator
        self.netD.load_state_dict(torch.load('./pretrained_models/netD.pth', map_location=lambda storage, loc: storage))
        for param in self.netD.parameters():
            param.requires_grad = False
    def learned_loss(self, hx, H, params=None):
        actions = H[:, 0:6]
        env_feature = H[:, 6:518]
        G_feature = H[:, 518:1030]
        env_score = self.netD(env_feature)
        criticG_fake = self.netD(G_feature)
        G_cost = -criticG_fake
        loss = actions*G_cost
        out = loss.sum(0).pow(2).sum(0).pow(0.5)
        return out
