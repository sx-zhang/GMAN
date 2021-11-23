from __future__ import division

import torch.nn as nn
from .basemodel import BaseModel
from .tcn import TemporalConvNet
import torch.nn.functional as F
import torch


class EMLV1(BaseModel):
    def __init__(self, args):
        super(EMLV1, self).__init__(args)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def learned_loss(self, hx, H, params=None):
        actions = H[:, 0:6]
        prev_s = H[:, 6:518]
        current_s = H[:, 518:1030]
        sim = self.cos(prev_s, current_s)
        sim_out = torch.lt(sim, 0.9985).int()
        b = sim_out.argmax(dim=0)
        if sim_out[b] == 0:
            loss = actions[b, :]
        else:
            effect_action = actions * sim_out.view(actions.shape[0], 1)
            loss = effect_action.sum(0)
        out = loss.pow(2).sum(0).pow(0.5)
        return out
