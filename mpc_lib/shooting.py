import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class ShootingMethod(object):

    def __init__(self, model, policy, T=10, lr=0.1):
        self.T = T
        self.model = model
        self.policy= policy

        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.lr = lr
        # self.K = torch.randn(T, self.action_dim, self.state_dim).to(self.device)
        self.u = torch.zeros(T, self.action_dim).to(self.device)
        # self.lam = torch.zeros(T, self.action_dim).to(self.device)
        # self.K.requires_grad = True
        self.u.requires_grad = True
        self.rho = 0.#1e-2

    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def update_constraint(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            i = 0
            for K, u, lam in zip(self.K, self.u, self.lam):
                mu, _ = self.policy(s)
                ubar = torch.mv(K, s.squeeze()) + u
                s, r = self.model.step(s, ubar.unsqueeze(0))
                self.lam[i] += self.rho * (torch.tanh(mu.squeeze()) - ubar)
                i += 1

    def update(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        cost = 0.
        rho_2 = self.rho/2.0
        for u in self.u:
            s, r = self.model.step(s, torch.tanh(u.unsqueeze(0)))
            cost = cost + r

        cost.backward()
        with torch.no_grad():
            self.u += self.lr * self.u.grad
            self.u.grad.zero_()
            u = torch.tanh(self.u[0].cpu().clone()).numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            return u
