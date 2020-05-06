import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class ShootingMethod(object):

    def __init__(self, model, T=10, lr=0.02):
        self.T = T
        self.model = model

        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.lr = lr
        self.u = torch.zeros(T, self.action_dim).to(self.device)
        self.u.requires_grad = True

        self.optim = optim.SGD([self.u], lr=lr)

    def reset(self):
        with torch.no_grad():
            self.u.zero_()


    def update(self, state, epochs=2):
        for epoch in range(epochs):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            for u in self.u:
                s, r = self.model.step(s, torch.tanh(u.unsqueeze(0)))
                cost = cost - r

            self.optim.zero_grad()
            cost.backward()
            self.optim.step()
        with torch.no_grad():
            u = torch.tanh(self.u[0].cpu().clone()).numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            return u
