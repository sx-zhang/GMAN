from __future__ import division

import torch.nn as nn
from .basemodelv0 import BaseModelv0
from .tcn import TemporalConvNet
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable


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


class EMLV3Base(BaseModelv0):
    def __init__(self, args):
        super(EMLV3Base, self).__init__(args)
        self.args = args
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.num_steps = args.num_steps
        discriminator = Discriminator_D1()
        self.netD = discriminator
        self.netD.load_state_dict(torch.load('./pretrained_models/netD.pth', map_location=lambda storage, loc: storage))
        for param in self.netD.parameters():
            param.requires_grad = False
        self.gammaD = 10
        self.lambda1 = 10
        # self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        # self.optimizerD = torch.optim.SGD(self.netD.parameters(), lr=0.001, momentum=0.9)
        # self.one = torch.tensor(1, dtype=torch.float)  # input label
        # self.mone = self.one * -1

    def learned_loss(self, hx, H, params=None):
        actions = H[:, 0:6]
        # env_feature = H[:, 6:518]
        G_feature = H[:, 518:1030]
        D_embedding = F.leaky_relu(
            F.linear(
                G_feature,
                weight=params["netD.fc1.weight"],
                bias=params["netD.fc1.bias"],
            ),
            negative_slope=0.2,
            inplace=True,
        )
        criticG_fake = F.linear(
            D_embedding,
            weight=params["netD.fc2.weight"],
            bias=params["netD.fc2.bias"],
        )
        G_cost = -criticG_fake
        loss = actions*G_cost
        out = loss.sum(0).pow(2).sum(0).pow(0.5)
        return out

    def D_loss(self, hx, H, params=None):
        env_feature = H[:, 6:518].detach()
        G_feature = H[:, 518:1030].detach()
        real_embedding = F.leaky_relu(
            F.linear(
                env_feature,
                weight=params["netD.fc1.weight"],
                bias=params["netD.fc1.bias"],
            ),
            negative_slope=0.2,
            inplace=True,
        )
        criticD_real = F.linear(
            real_embedding,
            weight=params["netD.fc2.weight"],
            bias=params["netD.fc2.bias"],
        )
        criticD_real = self.gammaD * criticD_real.mean()

        fake_embedding = F.leaky_relu(
            F.linear(
                G_feature,
                weight=params["netD.fc1.weight"],
                bias=params["netD.fc1.bias"],
            ),
            negative_slope=0.2,
            inplace=True,
        )
        criticD_fake = F.linear(
            fake_embedding,
            weight=params["netD.fc2.weight"],
            bias=params["netD.fc2.bias"],
        )
        criticD_fake = self.gammaD * criticD_fake.mean()

        gradient_penalty = self.gammaD * self.calc_gradient_penalty(params, env_feature, G_feature)

        out = -1*criticD_real + criticD_fake + gradient_penalty
        return out

    def calc_gradient_penalty(self, params, real_data, fake_data):
        alpha = torch.rand(self.num_steps, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(torch.device(real_data.device))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        # disc_interpolates = netD(interpolates)
        interpolates_embedding = F.leaky_relu(
            F.linear(
                interpolates,
                weight=params["netD.fc1.weight"],
                bias=params["netD.fc1.bias"],
            ),
            negative_slope=0.2,
            inplace=True,
        )
        disc_interpolates = F.linear(
            interpolates_embedding,
            weight=params["netD.fc2.weight"],
            bias=params["netD.fc2.bias"],
        )
        ones = torch.ones(disc_interpolates.size())
        ones = ones.to(torch.device(real_data.device))
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=ones,
                                        create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda1
        return gradient_penalty